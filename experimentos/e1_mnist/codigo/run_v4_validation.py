"""
Runner de validação: V4 h128/s2/g8/no-skip vs MLP 128
10 seeds, 10 épocas, com curvas de treinamento completas.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "resultados_finais" / "e1_mnist_validation"
EXPERIMENT = ROOT / "experimentos" / "e1_mnist" / "codigo" / "e1_mnist_v4.py"

CONFIGS = [
    {"model": "dense", "hidden": 128, "states": 0, "gate_hidden": 0, "skip": False},
    {"model": "v4", "hidden": 128, "states": 2, "gate_hidden": 8, "skip": False},
]


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def run_command(args: list[str]) -> None:
    print(" ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def load_result(path: Path, model: str) -> dict | None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = "dense" if model == "dense" else "v4_sparse"
    row = data.get(key)
    if not row:
        return None
    history = row.get("history", [])
    final_loss = history[-1]["loss"] if history else None

    flops = row.get("estimated_inference_flops_per_sample" if model == "dense"
                    else "estimated_sparse_inference_flops_per_sample", 0)
    return {
        "model": model,
        "seed": data["seed"],
        "hidden": row["hidden"],
        "states": row.get("states", 0) if model != "dense" else 0,
        "gate_hidden": row.get("gate_hidden", 0) if model != "dense" else 0,
        "use_skip": row.get("use_skip", False) if model != "dense" else False,
        "accuracy": row["accuracy"],
        "final_loss": final_loss,
        "train_time_sec": row.get("train_time_sec", 0),
        "inference_time_sec": row.get("inference_time_sec", 0),
        "flops_per_sample": flops,
        "params": row.get("params", 0),
        "accuracy_per_mflop": row["accuracy"] / (flops / 1_000_000) if flops else 0,
        "history": json.dumps(history),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed in seeds:
        for cfg in CONFIGS:
            if cfg["model"] == "dense":
                out_name = f"dense_h{cfg['hidden']}_seed{seed}.json"
                out_path = RESULT_DIR / out_name
                if args.skip_existing and out_path.exists():
                    print(f"  Pulando {out_name} (já existe)")
                else:
                    cmd = [
                        sys.executable, str(EXPERIMENT),
                        "--model", "dense",
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--hidden", str(cfg["hidden"]),
                        "--seed", str(seed),
                        "--lr", str(args.lr),
                        "--l2", str(args.l2),
                        "--out", f"e1_mnist_validation/{out_name}",
                    ]
                    run_command(cmd)
                row = load_result(out_path, "dense")
                if row:
                    rows.append(row)
            else:
                out_name = f"v4_h{cfg['hidden']}_s{cfg['states']}_g{cfg['gate_hidden']}_noskip_seed{seed}.json"
                out_path = RESULT_DIR / out_name
                if args.skip_existing and out_path.exists():
                    print(f"  Pulando {out_name} (já existe)")
                else:
                    cmd = [
                        sys.executable, str(EXPERIMENT),
                        "--model", "v4",
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--v4-hidden", str(cfg["hidden"]),
                        "--states", str(cfg["states"]),
                        "--gate-hidden", str(cfg["gate_hidden"]),
                        "--seed", str(seed),
                        "--lr", str(args.lr),
                        "--l2", str(args.l2),
                        "--no-skip",
                        "--out", f"e1_mnist_validation/{out_name}",
                    ]
                    run_command(cmd)
                row = load_result(out_path, "v4")
                if row:
                    rows.append(row)

    # CSV runs
    csv_path = RESULT_DIR / "runs.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Runs salvos em {csv_path}")

    # Summary
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["model"], row["hidden"], row["states"], row.get("gate_hidden", 0), row.get("use_skip", False))
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, items in sorted(grouped.items()):
        model, hidden, states, gate_hidden, use_skip = key
        n = len(items)
        accs = [i["accuracy"] for i in items]
        flops = items[0]["flops_per_sample"]
        params = items[0]["params"]
        summary.append({
            "model": model, "hidden": hidden, "states": states,
            "gate_hidden": gate_hidden, "use_skip": use_skip,
            "seeds": n,
            "accuracy_mean": sum(accs) / n,
            "accuracy_std": (sum((a - sum(accs)/n)**2 for a in accs) / n) ** 0.5 if n > 1 else 0,
            "accuracy_min": min(accs), "accuracy_max": max(accs),
            "final_loss_mean": sum(i["final_loss"] for i in items) / n,
            "train_time_mean_sec": sum(i["train_time_sec"] for i in items) / n,
            "inference_time_mean_sec": sum(i["inference_time_sec"] for i in items) / n,
            "flops_per_sample": flops, "params": params,
            "accuracy_per_mflop_mean": sum(i["accuracy_per_mflop"] for i in items) / n,
        })

    summary_path = RESULT_DIR / "summary.csv"
    if summary:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Sumário salvo em {summary_path}")

    # Summary JSON
    with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": rows, "summary": summary}, f, indent=2)

    print("\n=== RESULTADO DA VALIDAÇÃO ===")
    for s in summary:
        print(f"  {s['model']:>5} h={s['hidden']:>3} s={s['states']:>2} "
              f"acc={s['accuracy_mean']*100:>6.2f}% ±{s['accuracy_std']*100:.2f} "
              f"[{s['accuracy_min']*100:.2f}-{s['accuracy_max']*100:.2f}] "
              f"flops={s['flops_per_sample']:>8} acc/MFLOP={s['accuracy_per_mflop_mean']:.4f}")

    wins_mlp = sum(1 for i in range(len(seeds)) if rows[i*2]["accuracy"] < rows[i*2+1]["accuracy"])
    wins_v4 = sum(1 for i in range(len(seeds)) if rows[i*2]["accuracy"] > rows[i*2+1]["accuracy"])
    ties = len(seeds) - wins_mlp - wins_v4
    print(f"\n  Confronto direto ({len(seeds)} seeds): MLP={wins_mlp} V4={wins_v4} Empates={ties}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
