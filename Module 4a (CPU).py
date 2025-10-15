#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 4a: Docking with AutoDock Vina (CPU)  [GRACEFUL STOP ENABLED]
- Reads VinaConfig.txt next to Vina binary (simple key=value file)
- Docks all prepared_ligands/*.pdbqt
- Writes:
    results/<id>_out.pdbqt       (atomic write)
    results/<id>_vina.log        (stdout/stderr capture)
    results/summary.csv
    results/leaderboard.csv
- Updates state/manifest.csv (vina_* fields)

Run:  python "Module 4.py"
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

# ---------------- Graceful Stop (Ctrl+C) ----------------
STOP_REQUESTED = False
HARD_STOP = False

def _handle_sigint(sig, frame):
    global STOP_REQUESTED, HARD_STOP
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n⏹️  Ctrl+C detected — finishing current ligand, then exiting cleanly...")
        print("   (Press Ctrl+C again to stop ASAP after a safe checkpoint.)")
    else:
        HARD_STOP = True
        print("\n⏭️  Second Ctrl+C — will abort the loop ASAP and finalize outputs.")

signal.signal(signal.SIGINT, _handle_sigint)

# ---------------- Paths ----------------
BASE = Path(".").resolve()
DIR_PREP = BASE / "prepared_ligands"
DIR_RESULTS = BASE / "results"
DIR_STATE = BASE / "state"
DIR_REC_FALLBACK = BASE / "receptors" / "target_prepared.pdbqt"

FILE_MANIFEST = DIR_STATE / "manifest.csv"
FILE_SUMMARY = DIR_RESULTS / "summary.csv"
FILE_LEADER = DIR_RESULTS / "leaderboard.csv"

for d in (DIR_RESULTS, DIR_STATE):
    d.mkdir(parents=True, exist_ok=True)

# -------------- Utilities --------------
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def read_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in headers})

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

# -------------- Manifest ---------------
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
    rows = read_csv(FILE_MANIFEST)
    out = {}
    for r in rows:
        row = {k: r.get(k,"") for k in MANIFEST_FIELDS}
        out[row["id"]] = row
    return out

def save_manifest(manifest: dict[str, dict]) -> None:
    rows = [{k: v.get(k,"") for k in MANIFEST_FIELDS} for _,v in sorted(manifest.items())]
    write_csv(FILE_MANIFEST, rows, MANIFEST_FIELDS)

# -------------- Config (from Vina dir) --------------
def find_vina_binary() -> Path:
    """
    Try common names in project root; otherwise require explicit path here if you prefer.
    """
    candidates = [
        BASE / "vina_1.2.7_win.exe",
        BASE / "vina.exe",
        BASE / "vina"
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise SystemExit("❌ Could not find Vina binary in project root (e.g., vina_1.2.7_win.exe). Place it next to this script.")

def parse_vina_config(cfg_path: Path) -> Dict[str, str]:
    """
    Parse key=value pairs; strip comments starting with '#'.
    """
    if not cfg_path.exists():
        raise SystemExit(f"❌ VinaConfig.txt not found next to Vina: {cfg_path}")
    conf: Dict[str, str] = {}
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        conf[k.strip().lower()] = v.strip()
    return conf

def as_float(d: Dict[str,str], k: str, default: float) -> float:
    try:
        return float(d.get(k, default))
    except Exception:
        return float(default)

def as_int(d: Dict[str,str], k: str, default: int) -> int:
    try:
        return int(str(d.get(k, default)).strip())
    except Exception:
        return int(default)

def load_runtime_config(vina_path: Path) -> tuple[dict, dict, Path, str]:
    """
    Returns: (box, vcfg, receptor_path, config_hash)
    - box: dict with center_x/center_y/center_z/size_x/size_y/size_z
    - vcfg: dict with exhaustiveness/num_modes/energy_range/(optional seed,cpu)
    - receptor_path: resolved Path (fallback to ./receptors/target_prepared.pdbqt)
    - config_hash: SHA1 of VinaConfig.txt
    """
    cfg_path = vina_path.parent / "VinaConfig.txt"
    conf = parse_vina_config(cfg_path)

    box = {
        "center_x": as_float(conf, "center_x", 0.0),
        "center_y": as_float(conf, "center_y", 0.0),
        "center_z": as_float(conf, "center_z", 0.0),
        "size_x":   as_float(conf, "size_x", 20.0),
        "size_y":   as_float(conf, "size_y", 20.0),
        "size_z":   as_float(conf, "size_z", 20.0),
    }
    vcfg = {
        "exhaustiveness": as_int(conf, "exhaustiveness", 8),
        "num_modes":      as_int(conf, "num_modes", 9),
        "energy_range":   as_int(conf, "energy_range", 3),
    }
    seed = conf.get("seed", "").strip()
    if seed:
        try: vcfg["seed"] = int(seed)
        except Exception: pass
    cpu = conf.get("cpu", "").strip()
    if cpu:
        try: vcfg["cpu"] = int(cpu)
        except Exception: pass

    rec_str = conf.get("receptor", "") or conf.get("receptor_file", "")
    if rec_str:
        rec = Path(rec_str)
        if not rec.is_absolute():
            rec = (vina_path.parent / rec).resolve()
    else:
        rec = DIR_REC_FALLBACK.resolve()
    if not rec.exists():
        raise SystemExit(f"❌ Receptor not found: {rec}")

    try:
        chash = hashlib.sha1((cfg_path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()[:10]
    except Exception:
        chash = "nohash"

    print("Vina binary:", str(vina_path))
    print("Using VinaConfig.txt:", str(cfg_path))
    print("Box:", box)
    print("Vina params:", vcfg)
    print("Receptor:", str(rec))

    return box, vcfg, rec, chash

# -------------- Vina helpers --------------
VINA_RESULT_RE = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)", re.I)

def vina_pose_is_valid(path: Path) -> tuple[bool, Optional[float]]:
    try:
        if not path.exists() or path.stat().st_size < 200:
            return (False, None)
        txt = path.read_text(errors="ignore")
        scores = [float(m.group(1)) for m in VINA_RESULT_RE.finditer(txt)]
        if not scores:
            return (False, None)
        return (True, min(scores))
    except Exception:
        return (False, None)

def run_vina(vina_cmd: Path, receptor: Path, ligand_pdbqt: Path,
             out_pose: Path, out_log: Path, box: dict, vcfg: dict) -> tuple[bool, str]:
    """
    Run Vina producing out_pose.tmp, then atomically rename to out_pose.
    We DO NOT pass --log (some builds lack it). We capture stdout/stderr.
    """
    ligand_pdbqt = ligand_pdbqt.resolve()
    out_pose = out_pose.resolve()
    out_pose.parent.mkdir(parents=True, exist_ok=True)
    tmp_pose = out_pose.with_suffix(".pdbqt.tmp")

    # Clean stale outputs
    for p in (out_pose, tmp_pose, out_log):
        try:
            if Path(p).exists(): Path(p).unlink()
        except Exception:
            pass

    cmd = [
        str(vina_cmd),
        "--receptor", str(receptor),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(vcfg.get("exhaustiveness", 8)),
        "--num_modes", str(vcfg.get("num_modes", 9)),
        "--energy_range", str(vcfg.get("energy_range", 3)),
        "--out", str(tmp_pose)
    ]
    if "seed" in vcfg:
        cmd += ["--seed", str(vcfg["seed"])]
    if "cpu" in vcfg:
        cmd += ["--cpu", str(vcfg["cpu"])]

    # Run Vina and capture output to our own log
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
        last = (err or out or f"Vina rc={proc.returncode}").strip().splitlines()[-1][:300]
        return (False, last)

    ok, _ = vina_pose_is_valid(tmp_pose)
    if not ok:
        try:
            if tmp_pose.exists(): tmp_pose.unlink()
        except Exception:
            pass
        last = (err or out or "Invalid/empty Vina pose").strip().splitlines()[-1][:300]
        return (False, last)

    tmp_pose.replace(out_pose)
    return (True, "OK")

# -------------- Summary builders --------------
def build_and_write_summaries_from_manifest(manifest: dict[str, dict]) -> None:
    # Summary
    summary_headers = ["id","inchikey","vina_score","pose_path","created_at"]
    summary_rows = []
    for _, m in sorted(manifest.items()):
        sc = m.get("vina_score","")
        if sc:
            summary_rows.append({
                "id": m.get("id",""),
                "inchikey": m.get("inchikey",""),
                "vina_score": sc,
                "pose_path": m.get("vina_pose",""),
                "created_at": m.get("updated_at","")
            })
    write_csv(FILE_SUMMARY, summary_rows, summary_headers)

    # Leaderboard
    leader_headers = ["rank","id","inchikey","vina_score","pose_path"]
    ranked = sorted(summary_rows, key=lambda r: float(r["vina_score"])) if summary_rows else []
    leader_rows = []
    for i, r in enumerate(ranked, 1):
        leader_rows.append({
            "rank": i,
            "id": r["id"],
            "inchikey": r["inchikey"],
            "vina_score": r["vina_score"],
            "pose_path": r["pose_path"]
        })
    write_csv(FILE_LEADER, leader_rows, leader_headers)

# -------------- Main --------------
def main():
    vina_bin = find_vina_binary()
    box, vcfg, receptor, chash = load_runtime_config(vina_bin)

    ligs = sorted(DIR_PREP.glob("*.pdbqt"))
    if not ligs:
        raise SystemExit("❌ No ligand PDBQTs found in prepared_ligands/. Run Module 3 first.")

    receptor_sha1 = sha1_of_file(receptor)
    manifest = load_manifest()
    created_ts = now_iso()
    done = failed = 0

    try:
        for idx, lig in enumerate(ligs, 1):
            if STOP_REQUESTED or HARD_STOP:
                print("🧾 Stop requested — finalizing after this checkpoint...")
                break

            lig_id = lig.stem
            out_pose = (DIR_RESULTS / f"{lig_id}_out.pdbqt").resolve()
            out_log  = (DIR_RESULTS / f"{lig_id}_vina.log").resolve()

            # ---- IDEMPOTENCY CHECKS ----
            m = manifest.get(lig_id, {k: "" for k in MANIFEST_FIELDS})
            m.setdefault("id", lig_id)
            m.setdefault("created_at", created_ts)

            has_valid_pose, best_existing = vina_pose_is_valid(out_pose)
            same_cfg = (m.get("config_hash") == chash) and (m.get("receptor_sha1") == receptor_sha1)
            already_done = (m.get("vina_status") == "DONE")

            if has_valid_pose and already_done and same_cfg:
                # Keep existing result; repair/ensure manifest fields
                if best_existing is not None and not m.get("vina_score"):
                    m["vina_score"] = f"{best_existing:.2f}"
                m["vina_pose"] = str(out_pose)
                m["pdbqt_path"] = str(lig.resolve())
                m["tools_vina"] = str(vina_bin)
                m["updated_at"] = now_iso()
                manifest[lig_id] = m

                # Minimal log for transparency
                out_log.parent.mkdir(parents=True, exist_ok=True)
                with open(out_log, "w", encoding="utf-8") as f:
                    f.write("[SKIP] Existing valid pose kept (same receptor+config)\n")

                # Periodic checkpoint
                if idx % 50 == 0:
                    save_manifest(manifest)
                    build_and_write_summaries_from_manifest(manifest)
                continue
            # ---- END IDEMPOTENCY CHECKS ----

            # Fresh docking (or re-docking due to changed config/receptor or invalid/missing pose)
            ok, reason = run_vina(vina_bin, receptor, lig, out_pose, out_log, box, vcfg)

            m["pdbqt_path"] = str(lig.resolve())
            m["vina_status"] = "DONE" if ok else "FAILED"
            m["vina_pose"] = str(out_pose)
            m["vina_reason"] = "OK" if ok else reason
            m["config_hash"] = chash
            m["receptor_sha1"] = receptor_sha1
            m["tools_vina"] = str(vina_bin)
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

            # periodic checkpoint
            if idx % 50 == 0:
                save_manifest(manifest)
                build_and_write_summaries_from_manifest(manifest)

    finally:
        # Always flush outputs (even on Ctrl+C/exception)
        save_manifest(manifest)
        build_and_write_summaries_from_manifest(manifest)
        print(f"✅ Docking complete (or stopped). DONE: {done}  FAILED: {failed}")
        print(f"   Summary: {FILE_SUMMARY}")
        print(f"   Leaderboard: {FILE_LEADER}")
        print(f"   Manifest updated: {FILE_MANIFEST}")
        if STOP_REQUESTED or HARD_STOP:
            print("   (Exited early by user request.)")

if __name__ == "__main__":
    main()
