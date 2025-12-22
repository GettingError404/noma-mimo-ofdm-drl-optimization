"""
export_results_snapshot_json.py

Creates a compact JSON snapshot of experiment performance
for easy comparison across runs.
"""

import os
import json
from datetime import datetime
import numpy as np

RESULTS_DIR = "results"
OUTPUT_DIR = "results_snapshots"
LAST_N_EPISODES = 100
TARGET_SNR_DB = 20


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_training(log_path):
    data = load_json(log_path)

    rewards = np.array(data.get("rewards", []))
    se = np.array(data.get("SE", []))
    ee = np.array(data.get("EE", []))

    if len(rewards) == 0:
        return {}

    tail = slice(-LAST_N_EPISODES, None)

    return {
        "episodes": len(rewards),
        "last_N": LAST_N_EPISODES,
        "reward_mean": float(np.mean(rewards[tail])),
        "reward_std": float(np.std(rewards[tail])),
        "SE_mean": float(np.mean(se[tail])),
        "EE_mean_MbJ": float(np.mean(ee[tail]) / 1e6),
        "reward_best": float(np.max(rewards)),
    }


def extract_eval_at_snr(eval_data, snr_target):
    snr_list = np.array(eval_data["SNR_dB"])
    idx = int(np.argmin(np.abs(snr_list - snr_target)))

    out = {}
    for algo, metrics in eval_data.items():
        if algo == "SNR_dB":
            continue
        out[algo] = {
            "SE_bpsHz": float(metrics["SE"][idx]),
            "EE_MbJ": float(metrics["EE"][idx] / 1e6),
        }

    out["actual_SNR_dB"] = float(snr_list[idx])
    return out


def export_snapshot():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now()
    fname = timestamp.strftime("snapshot_%Y%m%d_%H%M%S.json")
    out_path = os.path.join(OUTPUT_DIR, fname)

    snapshot = {
        "timestamp": timestamp.isoformat(),
        "system": "NOMA-MIMO-OFDM DRL (TD3)",
        "training_summary": {},
        "evaluation_SNR20dB": {},
        "figures": [],
    }

    # -------- Training summary --------
    logs_dir = os.path.join(RESULTS_DIR, "training_logs")
    if os.path.exists(logs_dir):
        log_files = sorted(f for f in os.listdir(logs_dir) if f.endswith(".json"))
        if log_files:
            snapshot["training_summary"] = summarize_training(
                os.path.join(logs_dir, log_files[-1])
            )

    # -------- Evaluation results --------
    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if os.path.exists(eval_path):
        eval_data = load_json(eval_path)
        snapshot["evaluation_SNR20dB"] = extract_eval_at_snr(
            eval_data, TARGET_SNR_DB
        )

    # -------- Figures --------
    figs_dir = os.path.join(RESULTS_DIR, "figures")
    if os.path.exists(figs_dir):
        snapshot["figures"] = sorted(os.listdir(figs_dir))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"✅ Compact JSON snapshot saved to:\n{out_path}")


if __name__ == "__main__":
    export_snapshot()
