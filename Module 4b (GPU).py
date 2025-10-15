#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 4b: Docking with Vina-GPU (GPU) — config-driven + graceful stop
- Reads VinaGPUConfig.txt (preferred) or VinaConfig.txt from the GPU exe folder
- Per-ligand invocation by default (graceful Ctrl+C stop), optional batch mode
- Writes:
    results/<id>_out.pdbqt
    results/<id>_vinagpu.log
    results/summary.csv
    results/leaderboard.csv
- Updates state/manifest.csv (vina_* fields)

Run:  python "Module 4b.py"
"""

from __future__ import annotations
import csv
import hashlib
import json
import re
import shlex
import signal
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

# ---------- Graceful Stop ----------
STOP_REQUESTED = False
HARD_STOP = False
def _handle_sigint(sig, frame):
    global STOP_REQUESTED, HARD_STOP
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n⏹️  Ctrl+C — finishing current ligand then exiting cleanly...")
        print("   (Press Ctrl+C again to stop ASAP after checkpoint.)")
    else:
        HARD_STOP = True
        print("\n⏭️  Second Ctrl+C — will stop ASAP and finalize outputs.")
signal.signal(signal.SIGINT, _handle_sigint)

# ---------- Paths ----------
BASE = Path(".").resolve()
DIR_PREP   = BASE / "prepared_ligands"
DIR_RESULTS= BASE / "results"
DIR_STATE  = BASE / "state"
DIR_REC_FALLBACK = BASE / "receptors" / "target_prepared.pdbqt"

FILE_MANIFEST = DIR_STATE / "manifest.csv"
FILE_SUMMARY  = DIR_RESULTS / "summary.csv"
FILE_LEADER   = DIR_RESULTS / "leaderboard.csv"

for d in (DIR_RESULTS, DIR_STATE):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Utils ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def read_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers); w.writeheader()
        for r in rows: w.writerow({k: r.get(k,"") for k in headers})

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

# ---------- Manifest ----------
MANIFEST_FIELDS = [
    "id","smiles","inchikey",
    "admet_status","admet_reason",
    "sdf_status","sdf_path","sdf_reason",
    "pdbqt_status","pdbqt_path","pdbqt_reason",
    "vina_status","vina_score","vina_pose","vina_reason",
    "config_hash","receptor_sha1","tools_rdkit","tools_meeko","tools_vina",
    "created_at","updated_at"
]
def load_manifest() -> dict[str, dict]:
    if not FILE_MANIFEST.exists(): return {}
    rows = read_csv(FILE_MANIFEST); out={}
    for r in rows:
        row = {k: r.get(k,"") for k in MANIFEST_FIELDS}
        out[row["id"]] = row
    return out
def save_manifest(manifest: dict[str, dict]) -> None:
    rows = [{k: v.get(k,"") for k in MANIFEST_FIELDS} for _,v in sorted(manifest.items())]
    write_csv(FILE_MANIFEST, rows, MANIFEST_FIELDS)

# ---------- Config discovery (GPU) ----------
def find_vinagpu_binary() -> Path:
    candidates = [
        BASE / "Vina-GPU+.exe",
        BASE / "Vina-GPU+_K.exe",
        BASE / "Vina-GPU.exe",
        BASE / "vina-gpu.exe",
        BASE / "vina-gpu"
    ]
    for c in candidates:
        if c.exists(): return c.resolve()
    raise SystemExit("❌ Vina-GPU binary not found in project root (e.g., Vina-GPU+.exe).")

def parse_cfg_file(cfg_path: Path) -> Dict[str,str]:
    if not cfg_path.exists():
        raise SystemExit(f"❌ Config file not found: {cfg_path}")
    conf: Dict[str,str] = {}
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        # allow "key = value" or "key=value"
        if "#" in line: line = line.split("#",1)[0].strip()
        if "=" not in line: continue
        k, v = line.split("=",1)
        conf[k.strip().lower()] = v.strip()
    return conf

def as_float(d: Dict[str,str], k: str, default: float) -> float:
    try: return float(d.get(k, default))
    except Exception: return float(default)
def as_int(d: Dict[str,str], k: str, default: int) -> int:
    try: return int(str(d.get(k, default)).strip())
    except Exception: return int(default)

def load_runtime_config(vgpu_path: Path) -> tuple[dict, dict, Path, str, Path, Path]:
    """
    Returns: (box, gcfg, receptor_path, config_hash, lig_dir, out_dir)
    - box: center_x/y/z, size_x/y/z
    - gcfg: thread (>=1000 for your build), search_depth
    - receptor_path: resolved path
    - config_hash: SHA1 of cfg text
    - lig_dir/out_dir: optional from config; fallbacks applied
    """
    # Prefer VinaGPUConfig.txt, fallback to VinaConfig.txt
    cfg_gpu = vgpu_path.parent / "VinaGPUConfig.txt"
    cfg_cpu = vgpu_path.parent / "VinaConfig.txt"
    cfg_path = cfg_gpu if cfg_gpu.exists() else cfg_cpu
    conf = parse_cfg_file(cfg_path)

    box = {
        "center_x": as_float(conf, "center_x", 0.0),
        "center_y": as_float(conf, "center_y", 0.0),
        "center_z": as_float(conf, "center_z", 0.0),
        "size_x":   as_float(conf, "size_x", 20.0),
        "size_y":   as_float(conf, "size_y", 20.0),
        "size_z":   as_float(conf, "size_z", 20.0),
    }
    gcfg = {
        "thread":       as_int(conf, "thread", 10000),   # some builds require >=1000
        "search_depth": as_int(conf, "search_depth", 32),
    }
    if gcfg["thread"] < 1000:
        gcfg["thread"] = 1000  # enforce minimum for your binary quirk

    # receptor
    rec_str = conf.get("receptor","") or conf.get("receptor_file","")
    rec = Path(rec_str) if rec_str else DIR_REC_FALLBACK
    if not rec.is_absolute(): rec = (vgpu_path.parent / rec).resolve()
    if not rec.exists(): raise SystemExit(f"❌ Receptor not found: {rec}")

    # ligand/output dirs from config (optional)
    lig_dir = Path(conf["ligand_directory"]).resolve() if "ligand_directory" in conf else DIR_PREP
    out_dir = Path(conf["output_directory"]).resolve() if "output_directory" in conf else DIR_RESULTS

    chash = hashlib.sha1((cfg_path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()[:10]

    print("Vina-GPU binary:", vgpu_path)
    print("Config file:", cfg_path)
    print("Box:", box)
    print("GPU params:", gcfg)
    print("Receptor:", rec)
    print("Ligand dir:", lig_dir)
    print("Output dir:", out_dir)

    return box, gcfg, rec, chash, lig_dir, out_dir

# ---------- Pose validation ----------
VINA_RESULT_RE = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", re.I)
def vina_pose_is_valid(path: Path) -> Tuple[bool, Optional[float]]:
    try:
        if not path.exists() or path.stat().st_size < 200:
            return (False, None)
        txt = path.read_text(errors="ignore")
        scores = [float(m.group(1)) for m in VINA_RESULT_RE.finditer(txt)]
        if not scores: return (False, None)
        return (True, min(scores))
    except Exception:
        return (False, None)

# ---------- GPU run helpers ----------
def run_vinagpu_single(vgpu_cmd: Path, receptor: Path, ligand_pdbqt: Path,
                       out_pose: Path, out_log: Path, box: dict, gcfg: dict) -> tuple[bool, str]:
    """
    Per-ligand call (lets us stop gracefully).
    We pass explicit flags (instead of --config) to avoid overriding issues.
    """
    ligand_pdbqt = ligand_pdbqt.resolve()
    out_pose = out_pose.resolve()
    out_pose.parent.mkdir(parents=True, exist_ok=True)
    tmp_pose = out_pose.with_suffix(".pdbqt.tmp")

    # Clean stale
    for p in (out_pose, tmp_pose, out_log):
        try:
            if Path(p).exists(): Path(p).unlink()
        except Exception:
            pass

    cmd = [
        str(vgpu_cmd),
        "--receptor", str(receptor),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--thread", str(gcfg["thread"]),
        "--search_depth", str(gcfg["search_depth"]),
        "--out", str(tmp_pose),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()

    with open(out_log, "w", encoding="utf-8") as f:
        f.write("[BOX]\n"
                f"center_x={box['center_x']} center_y={box['center_y']} center_z={box['center_z']}\n"
                f"size_x={box['size_x']} size_y={box['size_y']} size_z={box['size_z']}\n\n")
        f.write("[CMD]\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
        f.write("\n[STDOUT]\n" + (out or "") + "\n")
        f.write("\n[STDERR]\n" + (err or "") + f"\nRC={proc.returncode}\n")

    if proc.returncode != 0:
        try:
            if tmp_pose.exists(): tmp_pose.unlink()
        except Exception:
            pass
        last = (err or out or f"Vina-GPU rc={proc.returncode}").strip().splitlines()[-1][:300]
        return (False, last)

    ok, _ = vina_pose_is_valid(tmp_pose)
    if not ok:
        try:
            if tmp_pose.exists(): tmp_pose.unlink()
        except Exception:
            pass
        last = (err or out or "Invalid/empty Vina-GPU pose").strip().splitlines()[-1][:300]
        return (False, last)

    tmp_pose.replace(out_pose)
    return (True, "OK")

def run_vinagpu_batch(vgpu_cmd: Path, receptor: Path, lig_dir: Path, out_dir: Path,
                      box: dict, gcfg: dict) -> int:
    """
    Folder mode (fast, but Ctrl+C will abort the whole batch process).
    Returns process return code.
    """
    cmd = [
        str(vgpu_cmd),
        "--config", str((vgpu_cmd.parent / "VinaGPUConfig.txt") if (vgpu_cmd.parent / "VinaGPUConfig.txt").exists() else (vgpu_cmd.parent / "VinaConfig.txt")),
        "--ligand_directory", str(lig_dir),
        "--output_directory", str(out_dir),
    ]
    # If your build ignores thread/search_depth in config, force them here:
    cmd += ["--thread", str(gcfg["thread"]), "--search_depth", str(gcfg["search_depth"])]

    print("Batch CMD:", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)

# ---------- Summaries ----------
def build_and_write_summaries_from_manifest(manifest: dict[str, dict]) -> None:
    # summary
    summary_headers = ["id","inchikey","vina_score","pose_path","created_at"]
    rows = []
    for _, m in sorted(manifest.items()):
        sc = m.get("vina_score","")
        if sc:
            rows.append({
                "id": m.get("id",""),
                "inchikey": m.get("inchikey",""),
                "vina_score": sc,
                "pose_path": m.get("vina_pose",""),
                "created_at": m.get("updated_at","")
            })
    write_csv(FILE_SUMMARY, rows, summary_headers)
    # leaderboard
    leader_headers = ["rank","id","inchikey","vina_score","pose_path"]
    ranked = sorted(rows, key=lambda r: float(r["vina_score"])) if rows else []
    leaders = []
    for i, r in enumerate(ranked, 1):
        leaders.append({"rank": i, **{k:r[k] for k in ("id","inchikey","vina_score","pose_path")}})
    write_csv(FILE_LEADER, leaders, leader_headers)

# ---------- Main ----------
def main():
    # toggle: per-ligand (graceful) vs batch
    use_batch = False  # set True if you want to try folder mode

    vgpu_bin = find_vinagpu_binary()
    box, gcfg, receptor, chash, lig_dir_cfg, out_dir_cfg = load_runtime_config(vgpu_bin)

    # Input ligands
    ligs = sorted((lig_dir_cfg if lig_dir_cfg.exists() else DIR_PREP).glob("*.pdbqt"))
    if not ligs:
        raise SystemExit("❌ No ligand PDBQTs found. Run Module 3 first.")

    # Outputs dir
    out_dir = out_dir_cfg if out_dir_cfg.exists() or not DIR_RESULTS.exists() else DIR_RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    receptor_sha1 = sha1_of_file(receptor)
    manifest = load_manifest()
    created_ts = now_iso()
    done = failed = 0

    if use_batch:
        # Fast path: one big process (no per-ligand logs/manifest updates mid-run)
        rc = run_vinagpu_batch(vgpu_bin, receptor, lig_dir_cfg, out_dir, box, gcfg)
        print(f"[Batch] Vina-GPU returned {rc}. You may need to parse outputs afterward.")
        # Optionally: parse produced *_out.pdbqt here to fill manifest
        # (omitted to keep batch path minimal)
        return

    # Per-ligand path (default; graceful stop + full logging)
    try:
        for idx, lig in enumerate(ligs, 1):
            if STOP_REQUESTED or HARD_STOP:
                print("🧾 Stop requested — finalizing after this checkpoint...")
                break

            lig_id = lig.stem
            out_pose = (out_dir / f"{lig_id}_out.pdbqt").resolve()
            out_log  = (out_dir / f"{lig_id}_vinagpu.log").resolve()

            ok, reason = run_vinagpu_single(vgpu_bin, receptor, lig, out_pose, out_log, box, gcfg)

            m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
            m["id"] = lig_id
            m["pdbqt_path"] = str(lig.resolve())
            m["vina_status"] = "DONE" if ok else "FAILED"
            m["vina_pose"] = str(out_pose)
            m["vina_reason"] = "OK" if ok else reason
            m["config_hash"] = chash
            m["receptor_sha1"] = receptor_sha1
            m["tools_vina"] = str(vgpu_bin)  # record GPU exe
            m.setdefault("created_at", created_ts)
            m["updated_at"] = now_iso()

            if ok:
                ok2, best_score = vina_pose_is_valid(out_pose)
                if ok2 and best_score is not None:
                    m["vina_score"] = f"{best_score:.2f}"
                    done += 1
                else:
                    m["vina_status"] = "FAILED"
                    m["vina_reason"] = "Pose written but invalid"
                    failed += 1
            else:
                failed += 1

            manifest[lig_id] = m

            # periodic checkpoint every 50
            if idx % 50 == 0:
                save_manifest(manifest)
                build_and_write_summaries_from_manifest(manifest)

    finally:
        # Always flush outputs
        save_manifest(manifest)
        build_and_write_summaries_from_manifest(manifest)
        print(f"✅ GPU docking complete (or stopped). DONE: {done}  FAILED: {failed}")
        print(f"   Summary: {FILE_SUMMARY}")
        print(f"   Leaderboard: {FILE_LEADER}")
        print(f"   Manifest updated: {FILE_MANIFEST}")
        if STOP_REQUESTED or HARD_STOP:
            print("   (Exited early by user request.)")

if __name__ == "__main__":
    main()
