#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 3: SDF -> ligand PDBQT via Meeko (robust, atomic writes)
- Inputs:
    - 3D_Structures/<id>.sdf   (from Module 2)
    - state/manifest.csv       (optional; used to map ids->sdf_path)
    - config/run.yml, config/machine.yml (optional; meeko_cmd, policy)
- Outputs:
    - prepared_ligands/<id>.pdbqt
    - prepared_ligands/<id>_meeko.log  (overwritten each run)
    - manifest updated (pdbqt_* fields, tools_meeko)

Run:  python m3_sdf_to_pdbqt_meeko.py
"""

from __future__ import annotations
import csv
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Try to read Meeko version for provenance
try:
    import meeko  # type: ignore
    MEEKO_VER = getattr(meeko, "__version__", "unknown")
except Exception:
    MEEKO_VER = ""

# Optional YAML
try:
    import yaml
except Exception:
    yaml = None

# ------------------------------ Paths ----------------------------------------
BASE = Path(".").resolve()
DIR_INPUT = BASE / "input"
DIR_STATE = BASE / "state"
DIR_SDF = BASE / "3D_Structures"
DIR_PDBQT = BASE / "prepared_ligands"
DIR_LOGS = BASE / "logs"

FILE_INPUT = DIR_INPUT / "input.csv"
FILE_MANIFEST = DIR_STATE / "manifest.csv"
FILE_RUNYML = BASE / "config" / "run.yml"
FILE_MACHINEYML = BASE / "config" / "machine.yml"

for d in (DIR_PDBQT, DIR_LOGS):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------ Helpers --------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]

def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})

def load_yaml(path: Path) -> dict:
    if not (yaml and path.exists()):
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v

def load_config() -> dict:
    # Defaults keep it simple and portable
    cfg = {
        "tools": {
            "meeko_cmd": "mk_prepare_ligand.py.exe",  # your chosen wrapper
            "python_exe": "python"                    # for fallback
        },
        "policy": {
            "skip_if_done": True
        }
    }
    deep_update(cfg, load_yaml(FILE_RUNYML))
    deep_update(cfg, load_yaml(FILE_MACHINEYML))
    return cfg

def config_hash() -> str:
    chunks = []
    for p in (FILE_RUNYML, FILE_MACHINEYML):
        if p.exists():
            chunks.append(p.read_text(encoding="utf-8"))
    if not chunks:
        chunks.append("{}")
    return hashlib.sha1("||".join(chunks).encode("utf-8")).hexdigest()[:10]

# ------------------------------ Manifest -------------------------------------
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
    if not FILE_MANIFEST.exists():
        return {}
    rows = read_csv(FILE_MANIFEST)
    out = {}
    for r in rows:
        row = {k: r.get(k, "") for k in MANIFEST_FIELDS}
        out[row["id"]] = row
    return out

def save_manifest(manifest: dict[str, dict]) -> None:
    rows = [{k: v.get(k, "") for k in MANIFEST_FIELDS} for _, v in sorted(manifest.items())]
    write_csv(FILE_MANIFEST, rows, MANIFEST_FIELDS)

# ------------------------------ Discovery ------------------------------------
def discover_sdf(manifest: dict[str, dict]) -> dict[str, Path]:
    """
    Prefer manifest sdf_path when present; else scan 3D_Structures.
    Return {id: sdf_path}
    """
    id2sdf: dict[str, Path] = {}
    # From manifest
    for lig_id, row in manifest.items():
        p = (row.get("sdf_path") or "").strip()
        if p:
            path = Path(p)
            if not path.is_absolute():
                path = (BASE / p).resolve()
            if path.exists():
                id2sdf[lig_id] = path
    # From folder (fill in missing)
    for sdf in sorted(DIR_SDF.glob("*.sdf")):
        lig_id = sdf.stem
        id2sdf.setdefault(lig_id, sdf.resolve())
    return id2sdf

# ------------------------------ Validation -----------------------------------
def pdbqt_is_valid(path: Path) -> bool:
    """
    Minimal ligand PDBQT validation:
    - file exists and non-trivial size
    - contains ATOM/HETATM records and a TORSDOF line
    """
    try:
        if not path.exists() or path.stat().st_size < 200:
            return False
        txt = path.read_text(errors="ignore")
        has_atom = ("ATOM " in txt) or ("HETATM" in txt)
        has_tors = "TORSDOF" in txt
        return has_atom and has_tors
    except Exception:
        return False

# ------------------------------ Meeko call -----------------------------------
def run_meeko(meeko_cmd: str, python_exe: str, in_sdf: Path, out_pdbqt: Path) -> tuple[bool, str, str]:
    """
    Try in order:
      1) mk_prepare_ligand.py.exe
      2) mk_prepare_ligand
      3) python -m meeko.cli_prepare_ligand
    Writes to out_pdbqt.tmp first, validates, then renames.
    Returns (ok, reason, log_path)
    """
    in_sdf = in_sdf.resolve()
    out_pdbqt = out_pdbqt.resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdbqt = out_pdbqt.with_suffix(".pdbqt.tmp")
    log_path = out_pdbqt.with_name(out_pdbqt.stem + "_meeko.log")

    # Clean stale outputs
    for p in (out_pdbqt, tmp_pdbqt, log_path):
        try:
            if Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass

    def _exec(cmd_list: list[str]) -> tuple[int, str, str]:
        proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        rc = proc.returncode
        Path(log_path).write_text(
            f">>> {' '.join(shlex.quote(c) for c in cmd_list)}\n\n[stdout]\n{out}\n\n[stderr]\n{err}\nRC={rc}\n",
            encoding="utf-8"
        )
        return rc, out, err

    # Candidate commands in order
    candidates = [
        [meeko_cmd, "-i", str(in_sdf), "-o", str(tmp_pdbqt)],                 # from YAML (default: mk_prepare_ligand.py.exe)
        ["mk_prepare_ligand", "-i", str(in_sdf), "-o", str(tmp_pdbqt)],       # plain mk_prepare_ligand
        [python_exe, "-m", "meeko.cli_prepare_ligand", "-i", str(in_sdf), "-o", str(tmp_pdbqt)],  # python -m fallback
    ]

    for cmd in candidates:
        try:
            rc, out, err = _exec(cmd)
            if rc == 0 and pdbqt_is_valid(tmp_pdbqt):
                tmp_pdbqt.replace(out_pdbqt)
                label = cmd[0]
                return True, f"OK via {label}", str(log_path)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    # Failure cleanup
    try:
        if tmp_pdbqt.exists():
            tmp_pdbqt.unlink()
    except Exception:
        pass
    return False, "All Meeko attempts failed", str(log_path)

# ------------------------------ Main -----------------------------------------
def main():
    cfg = load_config()
    chash = config_hash()
    meeko_cmd = str(cfg.get("tools", {}).get("meeko_cmd", "mk_prepare_ligand.py.exe"))
    python_exe = str(cfg.get("tools", {}).get("python_exe", "python"))
    skip_if_done = bool(cfg.get("policy", {}).get("skip_if_done", True))

    manifest = load_manifest()
    id2sdf = discover_sdf(manifest)

    if not id2sdf:
        raise SystemExit("❌ No SDFs found. Run Module 2 first.")

    done, failed = 0, 0
    created_ts = now_iso()

    for lig_id, sdf_path in sorted(id2sdf.items()):
        out_pdbqt = (DIR_PDBQT / f"{lig_id}.pdbqt").resolve()
        log_path = out_pdbqt.with_name(out_pdbqt.stem + "_meeko.log")

        # Skip only if an existing PDBQT validates
        if skip_if_done and pdbqt_is_valid(out_pdbqt):
            m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
            m["id"] = lig_id
            m["sdf_path"] = str(sdf_path)
            m["pdbqt_status"] = "DONE"
            m["pdbqt_path"] = str(out_pdbqt)
            m["pdbqt_reason"] = "Found existing valid PDBQT"
            m["config_hash"] = chash
            m["tools_meeko"] = MEEKO_VER or "Meeko"
            m.setdefault("created_at", created_ts)
            m["updated_at"] = now_iso()
            manifest[lig_id] = m
            # Overwrite log noting skip
            Path(log_path).write_text(f"[SKIP] Existing valid PDBQT kept: {out_pdbqt}\n", encoding="utf-8")
            done += 1
            continue

        ok, reason, _ = run_meeko(meeko_cmd, python_exe, sdf_path, out_pdbqt)

        # Update manifest
        m = manifest.get(lig_id, {k:"" for k in MANIFEST_FIELDS})
        m["id"] = lig_id
        m["sdf_path"] = str(sdf_path)
        m["pdbqt_status"] = "DONE" if ok else "FAILED"
        m["pdbqt_path"] = str(out_pdbqt)
        m["pdbqt_reason"] = "OK" if ok else reason
        m["config_hash"] = chash
        m["tools_meeko"] = MEEKO_VER or "Meeko"
        m.setdefault("created_at", created_ts)
        m["updated_at"] = now_iso()
        manifest[lig_id] = m

        if ok:
            done += 1
        else:
            failed += 1

    save_manifest(manifest)
    print(f"✅ SDF → PDBQT complete. DONE: {done}  FAILED: {failed}")
    print(f"   Outputs in: {DIR_PDBQT}")
    print(f"   Manifest updated: {FILE_MANIFEST}")

if __name__ == "__main__":
    main()
