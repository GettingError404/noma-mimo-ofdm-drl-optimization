"""
export_results_snapshot_compact.py

Exports FINAL, COMPACT, COPY-PASTE-FRIENDLY experiment summaries
into a JSONL file (one experiment = one line).
"""

import os
import json
from datetime import datetime
import numpy as np

RESULTS_DIR = "results"
OUTPUT_DIR = "results_snapshots"
OUTPUT_FILE = "experiments_compact.jsonl"


def safe_mean(arr):
    return float(np.mean(arr)) if arr else None


def safe_std(arr):
    return float(np.std(arr)) if arr else None


def export_compact_snapshot():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if not os.path.exists(eval_path):
        print("❌ evaluation_results.json not found")
        return

    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    timestamp = datetime.utcnow().isoformat()

    # ---- extract only what matters ----
    record = {
        "timestamp": timestamp,
        "scenario": eval_data.get("scenario"),
        "users": eval_data.get("K"),
        "Nt": eval_data.get("N_t"),
        "bandwidth_MHz": eval_data.get("bandwidth_MHz"),
        "psi": eval_data.get("psi"),
    }

    # Per-algorithm summary
    summaries = []
    for algo, results in eval_data.get("results", {}).items():
        summaries.append({
            "algo": algo,
            "snr_db": results.get("snr_db"),
            "SE_mean": safe_mean(results.get("SE")),
            "SE_std": safe_std(results.get("SE")),
            "EE_mean": safe_mean(results.get("EE")),
            "EE_std": safe_std(results.get("EE")),
        })

    record["summaries"] = summaries

    # ---- write as SINGLE LINE JSON ----
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"✅ Compact snapshot appended → {out_path}")


if __name__ == "__main__":
    export_compact_snapshot()
