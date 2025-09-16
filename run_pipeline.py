#!/usr/bin/env python3
# Automated Virtual Screening Orchestrator
# - Runs your module scripts in order (Module 1.py to Module 7.py if present)
# - Captures RAW and emoji-sanitized logs
# - Checkpoints so you can --resume long runs
# - Works on Windows (UTF-8 forced), macOS, Linux

import os
import sys
from pathlib import Path
import subprocess
import argparse
import datetime
from utils_text import sanitize_for_csv

DEFAULT_MODULE_PATTERNS = [
    "Module 1*.py",  # ADMET filtering
    "Module 2*.py",  # SMILES -> SDF (3D)
    "Module 3*.py",  # Ligand prep (Meeko)
    "Module 4*.py",  # Receptor prep
]

def find_modules(project_dir: Path, patterns):
    found = []
    for pat in patterns:
        matches = sorted(project_dir.glob(pat))
        found.append(matches[0] if matches else None)
    return found

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd_list, workdir: Path, env: dict):
    return subprocess.run(
        cmd_list,
        cwd=str(workdir),
        text=True,
        capture_output=True,
        env=env,
        shell=False
    )

def write_logs(stage_idx: int, name: str, out_dir: Path, stdout: str, stderr: str):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    ensure_dir(raw_dir)
    ensure_dir(clean_dir)

    # RAW logs
    (raw_dir / f"{stage_idx:02d}_{name}_{ts}.stdout.log").write_text(stdout or "", encoding="utf-8", errors="ignore")
    (raw_dir / f"{stage_idx:02d}_{name}_{ts}.stderr.log").write_text(stderr or "", encoding="utf-8", errors="ignore")

    # CLEAN logs (emoji-free)
    (clean_dir / f"{stage_idx:02d}_{name}_{ts}.stdout.log").write_text(sanitize_for_csv(stdout or ""), encoding="utf-8", errors="ignore")
    (clean_dir / f"{stage_idx:02d}_{name}_{ts}.stderr.log").write_text(sanitize_for_csv(stderr or ""), encoding="utf-8", errors="ignore")

def mark_done(stage_file: Path):
    stage_file.write_text(datetime.datetime.now().isoformat(), encoding="utf-8")

def is_done(stage_file: Path) -> bool:
    return stage_file.exists()

def main():
    ap = argparse.ArgumentParser(description="Automated Virtual Screening Orchestrator")
    ap.add_argument("--project", type=str, default=".", help="Folder with module scripts (default: current)")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to use")
    ap.add_argument("--resume", action="store_true", help="Resume from last unfinished stage")
    ap.add_argument("--fresh", action="store_true", help="Ignore checkpoints and re-run everything")
    ap.add_argument("--only", type=str, help="Run only stages (comma-separated 1-based indices)")
    ap.add_argument("--skip", type=str, help="Skip stages (comma-separated 1-based indices)")
    ap.add_argument("--list", action="store_true", help="List detected modules and exit")
    args = ap.parse_args()

    project_dir = Path(args.project).resolve()
    logs_dir = project_dir / "_pipeline_logs"
    ckpt_dir = project_dir / "_pipeline_checkpoints"
    ensure_dir(logs_dir)
    ensure_dir(ckpt_dir)

    modules = find_modules(project_dir, DEFAULT_MODULE_PATTERNS)

    if args.list:
        for i, m in enumerate(modules, start=1):
            print(f"{i}. {m.name if m else '— not found —'}")
        return

    only = set(int(i.strip()) for i in args.only.split(",")) if args.only else None
    skip = set(int(i.strip()) for i in args.skip.split(",")) if args.skip else set()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    failures = []

    for idx, module_path in enumerate(modules, start=1):
        if only and idx not in only:
            continue
        if idx in skip:
            print(f"Skipping stage {idx}")
            continue
        if module_path is None:
            print(f"Stage {idx}: module file not found — continuing.")
            continue

        ck_file = ckpt_dir / f"{idx:02d}.done"
        if args.resume and is_done(ck_file) and not args.fresh:
            print(f"Stage {idx} already completed — resume mode, skipping.")
            continue
        if args.fresh and is_done(ck_file):
            try:
                ck_file.unlink()
            except Exception:
                pass

        print(f"Running stage {idx}: {module_path.name}")
        cmd = [args.python, str(module_path)]
        proc = run_cmd(cmd, project_dir, env)

        stage_name = module_path.stem.replace(" ", "_")
        write_logs(idx, stage_name, logs_dir, proc.stdout, proc.stderr)

        if proc.returncode == 0:
            print(f"Stage {idx} finished OK.")
            mark_done(ck_file)
        else:
            print(f"Stage {idx} FAILED (exit {proc.returncode}). See logs in {logs_dir}")
            failures.append((idx, module_path.name, proc.returncode))
            break

    if failures:
        print("\nSummary: Some stages failed:")
        for i, name, code in failures:
            print(f"  - Stage {i}: {name} (exit {code})")
        sys.exit(1)
    else:
        print("\nPipeline completed successfully. Logs written to:", logs_dir)

if __name__ == "__main__":
    main()
