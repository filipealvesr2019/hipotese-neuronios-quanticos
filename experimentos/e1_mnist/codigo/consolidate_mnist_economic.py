"""
Consolida todos os JSONs existentes em resultados_finais/e1_mnist_economic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "resultados_finais" / "e1_mnist_economic"


def load_result(path: Path) -> dict | None:
    if path.name == "summary.json":
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "dense" in data:
        row = data["dense"]
        model = "dense"
        states = 0
        gate_hidden = 0
        use_skip = False
        flops = row["estimated_inference_flops_per_sample"]
    elif "v4_sparse" in data:
        row = data["v4_sparse"]
        model = "v4_economic"
        states = row["states"]
        gate_hidden = row["gate_hidden"]
        use_skip = row["use_skip"]
        flops = row["estimated_sparse_inference_flops_per_sample"]
    else:
        return None

    history = row["history"]
    final_loss = history[-1]["loss"] if history else None
    return {
        "model": model,
        "seed": data["seed"],
        "hidden": row["hidden"],
        "states": states,
        "gate_hidden": gate_hidden,
        "use_skip": use_skip,
        "accuracy": row["accuracy"],
        "final_loss": final_loss,
        "train_time_sec": row["train_time_sec"],
        "inference_time_sec": row["inference_time_sec"],
        "flops_per_sample": flops,
        "params": row["params"],
        "accuracy_per_mflop": row["accuracy"] / (flops / 1_000_000),
        "result_file": str(path.relative_to(ROOT)),
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["model"], row["hidden"], row["states"], row["gate_hidden"], row["use_skip"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (model, hidden, states, gate_hidden, use_skip), items in sorted(grouped.items()):
        n = len(items)
        summary.append({
            "model": model,
            "hidden": hidden,
            "states": states,
            "gate_hidden": gate_hidden,
            "use_skip": use_skip,
            "seeds": n,
            "accuracy_mean": sum(i["accuracy"] for i in items) / n,
            "final_loss_mean": sum(i["final_loss"] for i in items) / n,
            "train_time_mean_sec": sum(i["train_time_sec"] for i in items) / n,
            "inference_time_mean_sec": sum(i["inference_time_sec"] for i in items) / n,
            "flops_per_sample": items[0]["flops_per_sample"],
            "params": items[0]["params"],
            "accuracy_per_mflop_mean": sum(i["accuracy_per_mflop"] for i in items) / n,
        })
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = []
    for path in sorted(RESULT_DIR.glob("*.json")):
        row = load_result(path)
        if row:
            rows.append(row)

    summary = summarize(rows)
    write_csv(RESULT_DIR / "runs.csv", rows)
    write_csv(RESULT_DIR / "summary.csv", summary)
    with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": rows, "summary": summary}, f, indent=2)

    print(f"Consolidados {len(rows)} runs em {RESULT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
