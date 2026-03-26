from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import csv

def make_run_dir(model_tag: str, target_tag: str = "w2_inlet_p") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).resolve().parent.parent / "runs" / model_tag / f"study_{target_tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def get_trial_dir(run_dir: Path, trial_number: int) -> Path:
    d = run_dir / f"trial_{trial_number:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)