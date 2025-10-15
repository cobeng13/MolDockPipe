#!/usr/bin/env python3
# Module 4b (GPU) — Mini-batch with idempotent resume + graceful stop + atom-type validation

from __future__ import annotations
import csv, hashlib, re, shlex, shutil, signal, subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Iterable

# ------------ Tunables ------------
BATCH_SIZE = 64
SAFE_RESUME = True
KEEP_TMP = False
# ----------------------------------

STOP_REQUESTED = False
HARD_STOP = False
def _sigint(_, __):
    global STOP_REQUESTED, HARD_STOP
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n⏹️  Ctrl+C — finishing current batch then exiting cleanly…")
    else:
        HARD_STOP = True
        print("\n⏭️  Second Ctrl+C — will exit ASAP after harvest.")
signal.signal(signal.SIGINT, _sigint)

BASE = Path(".").resolve()
DIR_PREP   = BASE / "prepared_ligands"
DIR_RESULTS= BASE / "results"
DIR_STATE  = BASE / "state"
DIR_REC_FALLBACK = BASE / "receptors" / "target_prepared.pdbqt"

FILE_MANIFEST = DIR_STATE / "manifest.csv"
FILE_SUMMARY  = DIR_RESULTS / "summary.csv"
FILE_LEADER   = DIR_RESULTS / "leaderboard.csv"

for d in (DIR_RESULTS, DIR_STATE): d.mkdir(parents=True, exist_ok=True)

def now_iso()->str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def read_csv(path: Path)->list[dict]:
    if not path.exists(): return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def write_csv(path: Path, rows: list[dict], headers: list[str])->None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers); w.writeheader()
        for r in rows: w.writerow({k: r.get(k,"") for k in headers})

def sha1_of_file(p: Path)->str:
    h=hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

MANIFEST_FIELDS = [
    "id","smiles","inchikey",
    "admet_status","admet_reason",
    "sdf_status","sdf_path","sdf_reason",
    "pdbqt_status","pdbqt_path","pdbqt_reason",
    "vina_status","vina_score","vina_pose","vina_reason",
    "config_hash","receptor_sha1","tools_rdkit","tools_meeko","tools_vina",
    "created_at","updated_at"
]
def load_manifest()->dict[str,dict]:
    if not FILE_MANIFEST.exists(): return {}
    out={}
    for r in read_csv(FILE_MANIFEST):
        row={k:r.get(k,"") for k in MANIFEST_FIELDS}; out[row["id"]]=row
    return out
def save_manifest(m:dict[str,dict])->None:
    rows=[{k:v.get(k,"") for k in MANIFEST_FIELDS} for _,v in sorted(m.items())]
    write_csv(FILE_MANIFEST, rows, MANIFEST_FIELDS)

# --- Atom-type validation ---
ALLOWED_AD4_TYPES = {
    "C","A","N","O","S","H","P","F","Cl","Br","I",
    "HD","NA","OA","SA",
    "Zn","Fe","Mg","Mn","Ca","Cu","Ni","Co","K","Na"
}

def get_pdbqt_atom_types(path: Path) -> set[str]:
    types=set()
    try:
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith(("ATOM","HETATM")):
                toks=line.split()
                if toks: types.add(toks[-1])
    except Exception:
        pass
    return types

def pdbqt_has_only_allowed_types(path: Path) -> tuple[bool,str]:
    ts=get_pdbqt_atom_types(path)
    bad=[t for t in ts if t not in ALLOWED_AD4_TYPES]
    if bad:
        return False,"Unsupported AD4 atom types: "+",".join(sorted(set(bad)))
    return True,"OK"

def find_vinagpu_binary()->Path:
    for name in ("Vina-GPU+.exe","Vina-GPU+_K.exe","Vina-GPU.exe","vina-gpu.exe","vina-gpu"):
        p = BASE / name
        if p.exists(): return p.resolve()
    raise SystemExit("❌ Vina-GPU binary not found in project root.")

def parse_cfg(path: Path)->Dict[str,str]:
    if not path.exists(): raise SystemExit(f"❌ Config not found: {path}")
    conf={}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line=raw.strip()
        if not line or line.startswith("#"): continue
        if "#" in line: line=line.split("#",1)[0].strip()
        if "=" not in line: continue
        k,v=line.split("=",1); conf[k.strip().lower()]=v.strip()
    return conf
def as_float(d:Dict[str,str],k:str,default:float)->float:
    try: return float(d.get(k,default))
    except: return float(default)
def as_int(d:Dict[str,str],k:str,default:int)->int:
    try: return int(str(d.get(k,default)).strip())
    except: return int(default)

def load_runtime(vgpu: Path):
    cfg_gpu = vgpu.parent/"VinaGPUConfig.txt"
    cfg_cpu = vgpu.parent/"VinaConfig.txt"
    cfg_path = cfg_gpu if cfg_gpu.exists() else cfg_cpu
    conf = parse_cfg(cfg_path)
    box = {
        "center_x": as_float(conf,"center_x",0.0),
        "center_y": as_float(conf,"center_y",0.0),
        "center_z": as_float(conf,"center_z",0.0),
        "size_x":   as_float(conf,"size_x",20.0),
        "size_y":   as_float(conf,"size_y",20.0),
        "size_z":   as_float(conf,"size_z",20.0),
    }
    gcfg = {
        "thread": max(1000, as_int(conf,"thread",10000)),
        "search_depth": as_int(conf,"search_depth",32),
    }
    rec_str = conf.get("receptor","") or conf.get("receptor_file","")
    rec = Path(rec_str) if rec_str else DIR_REC_FALLBACK
    if not rec.is_absolute(): rec = (vgpu.parent/rec).resolve()
    if not rec.exists(): raise SystemExit(f"❌ Receptor not found: {rec}")

    lig_dir = Path(conf["ligand_directory"]).resolve() if "ligand_directory" in conf else DIR_PREP
    out_dir = Path(conf["output_directory"]).resolve() if "output_directory" in conf else DIR_RESULTS

    chash = hashlib.sha1((cfg_path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()[:10]
    print("Vina-GPU:", vgpu, "| Config:", cfg_path)
    print("Box:", box, "| GPU params:", gcfg)
    print("Ligand dir:", lig_dir, "| Output dir:", out_dir)
    return box,gcfg,rec,chash,lig_dir,out_dir,cfg_path

# --- Pose parsing ---
RES_RE = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", re.I)
def vina_pose_is_valid(p:Path)->Tuple[bool,Optional[float]]:
    try:
        if not p.exists() or p.stat().st_size<200: return (False,None)
        txt=p.read_text(errors="ignore")
        scores=[float(m.group(1)) for m in RES_RE.finditer(txt)]
        return ((len(scores)>0), (min(scores) if scores else None))
    except: return (False,None)

# --- Helpers ---
def chunked(it: Iterable[Path], n:int)->Iterable[list[Path]]:
    buf=[]
    for x in it:
        buf.append(x)
        if len(buf)==n:
            yield buf; buf=[]
    if buf: yield buf

def build_and_write_summaries(manifest: dict[str,dict])->None:
    summ_headers=["id","inchikey","vina_score","pose_path","created_at"]
    rows=[]
    for _,m in sorted(manifest.items()):
        sc=m.get("vina_score","")
        if sc:
            rows.append({"id":m.get("id",""),
                         "inchikey":m.get("inchikey",""),
                         "vina_score":sc,
                         "pose_path":m.get("vina_pose",""),
                         "created_at":m.get("updated_at","")})
    write_csv(FILE_SUMMARY, rows, summ_headers)
    lead_headers=["rank","id","inchikey","vina_score","pose_path"]
    ranked=sorted(rows, key=lambda r: float(r["vina_score"])) if rows else []
    leaders=[{"rank":i,"id":r["id"],"inchikey":r["inchikey"],
              "vina_score":r["vina_score"],"pose_path":r["pose_path"]}
             for i,r in enumerate(ranked,1)]
    write_csv(FILE_LEADER, leaders, lead_headers)

def run_batch(vgpu:Path, cfg_file:Path, lig_dir:Path, out_dir:Path, gcfg:dict)->int:
    cmd=[str(vgpu),"--config",str(cfg_file),
         "--ligand_directory",str(lig_dir),
         "--output_directory",str(out_dir),
         "--thread",str(gcfg["thread"]),
         "--search_depth",str(gcfg["search_depth"])]
    print("Batch CMD:", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)

# --- Main ---
def main():
    vgpu = find_vinagpu_binary()
    box,gcfg,receptor,chash,lig_dir,out_dir,cfg_file = load_runtime(vgpu)

    all_ligs = sorted(lig_dir.glob("*.pdbqt"))
    if not all_ligs: raise SystemExit("❌ No ligand PDBQTs found.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if SAFE_RESUME:
        pending=[p for p in all_ligs if not (out_dir / f"{p.stem}_out.pdbqt").exists()]
    else:
        pending=list(all_ligs)

    manifest = load_manifest()
    created_ts = now_iso()
    receptor_sha = sha1_of_file(receptor)

    # Filter out ligands with invalid atom types
    valid_pending=[]
    for lig in pending:
        ok, why = pdbqt_has_only_allowed_types(lig)
        if not ok:
            lig_id = lig.stem
            m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
            m["id"]=lig_id
            m["pdbqt_path"]=str(lig.resolve())
            m["vina_status"]="FAILED"
            m["vina_reason"]=why
            m.setdefault("created_at", created_ts)
            m["updated_at"]=now_iso()
            manifest[lig_id]=m
            print(f"⚠️ Skipping {lig.name} — {why}")
        else:
            valid_pending.append(lig)
    save_manifest(manifest)

    if not valid_pending:
        print("✅ No valid ligands left to process. Summaries updated.")
        build_and_write_summaries(manifest)
        return

    # Mini-batch loop
    tmp_root = out_dir / "_batch_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        for bi, batch in enumerate(chunked(valid_pending, BATCH_SIZE), 1):
            if STOP_REQUESTED or HARD_STOP:
                print("🧾 Stop requested — exiting before next batch.")
                break

            tmp_dir = tmp_root / f"b{bi:04d}"
            if tmp_dir.exists(): shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for lig in batch: shutil.copy2(lig, tmp_dir / lig.name)

            rc = run_batch(vgpu, cfg_file, tmp_dir, out_dir, gcfg)
            if rc != 0:
                print(f"⚠️ Batch {bi} rc={rc}. Harvesting outputs then stopping.")

            # Harvest
            for lig in batch:
                lig_id = lig.stem
                pose = out_dir / f"{lig_id}_out.pdbqt"
                ok,best = vina_pose_is_valid(pose)
                m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
                m["id"]=lig_id
                m["pdbqt_path"]=str(lig.resolve())
                m["vina_status"]="DONE" if ok else "FAILED"
                m["vina_pose"]=str(pose.resolve())
                m["vina_reason"]="OK" if ok else "No VINA RESULT found"
                m["vina_score"]=f"{best:.2f}" if ok and best is not None else ""
                m["config_hash"]=chash
                m["receptor_sha1"]=receptor_sha
                m["tools_vina"]=str(vgpu)
                m.setdefault("created_at", created_ts)
                m["updated_at"]=now_iso()
                manifest[lig_id]=m

            save_manifest(manifest)
            build_and_write_summaries(manifest)
            if not KEEP_TMP:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if rc != 0:
                break

    finally:
        if not KEEP_TMP:
            shutil.rmtree(tmp_root, ignore_errors=True)
        save_manifest(manifest)
        build_and_write_summaries(manifest)
        print("✅ Mini-batch GPU docking done (or safely stopped).")
        print(f"Manifest: {FILE_MANIFEST}")
        print(f"Summary : {FILE_SUMMARY}")
        print(f"Leaders : {FILE_LEADER}")

if __name__ == "__main__":
    main()
