"""
Runner da matriz E1/E2 para MNIST.

Exemplo rapido:
  python experimentos/e1_mnist/codigo/run_mnist_matrix.py --seeds 1 --epochs 1 --train-limit 1000 --test-limit 500

Experimento planejado:
  python experimentos/e1_mnist/codigo/run_mnist_matrix.py --seeds 1-10 --epochs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "resultados_finais" / "e1_mnist_matrix"
EXPERIMENT = ROOT / "experimentos" / "e1_mnist" / "codigo" / "e1_mnist_v4.py"


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


def load_result(path: Path, model: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = "dense" if model == "dense" else "v4_sparse"
    row = data[key]
    history = row["history"]
    final_loss = history[-1]["loss"] if history else None
    if model == "dense":
        flops = row["estimated_inference_flops_per_sample"]
        hidden = row["hidden"]
        states = 0
        gate_hidden = 0
    else:
        flops = row["estimated_sparse_inference_flops_per_sample"]
        hidden = row["hidden"]
        states = row["states"]
        gate_hidden = row["gate_hidden"]
    return {
        "model": model,
        "seed": data["seed"],
        "hidden": hidden,
        "states": states,
        "gate_hidden": gate_hidden,
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
        key = (row["model"], row["hidden"], row["states"], row["gate_hidden"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (model, hidden, states, gate_hidden), items in sorted(grouped.items()):
        n = len(items)
        summary.append({
            "model": model,
            "hidden": hidden,
            "states": states,
            "gate_hidden": gate_hidden,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dense-hidden", default="64,128,256")
    parser.add_argument("--v4-hidden", default="64,128,256")
    parser.add_argument("--states", default="4")
    parser.add_argument("--gate-hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    dense_hidden = parse_int_list(args.dense_hidden)
    v4_hidden = parse_int_list(args.v4_hidden)
    states_list = parse_int_list(args.states)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed in seeds:
        for hidden in dense_hidden:
            out_name = f"dense_h{hidden}_seed{seed}.json"
            out_path = RESULT_DIR / out_name
            if not (args.skip_existing and out_path.exists()):
                cmd = [
                    sys.executable,
                    str(EXPERIMENT),
                    "--model", "dense",
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--hidden", str(hidden),
                    "--seed", str(seed),
                    "--lr", str(args.lr),
                    "--l2", str(args.l2),
                    "--out", f"e1_mnist_matrix/{out_name}",
                ]
                if args.train_limit:
                    cmd.extend(["--train-limit", str(args.train_limit)])
                if args.test_limit:
                    cmd.extend(["--test-limit", str(args.test_limit)])
                run_command(cmd)
            rows.append(load_result(out_path, "dense"))

        for states in states_list:
            for hidden in v4_hidden:
                out_name = f"v4_h{hidden}_s{states}_seed{seed}.json"
                out_path = RESULT_DIR / out_name
                if not (args.skip_existing and out_path.exists()):
                    cmd = [
                        sys.executable,
                        str(EXPERIMENT),
                        "--model", "v4",
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--hidden", "128",
                        "--v4-hidden", str(hidden),
                        "--states", str(states),
                        "--gate-hidden", str(args.gate_hidden),
                        "--seed", str(seed),
                        "--lr", str(args.lr),
                        "--l2", str(args.l2),
                        "--out", f"e1_mnist_matrix/{out_name}",
                    ]
                    if args.train_limit:
                        cmd.extend(["--train-limit", str(args.train_limit)])
                    if args.test_limit:
                        cmd.extend(["--test-limit", str(args.test_limit)])
                    run_command(cmd)
                rows.append(load_result(out_path, "v4"))

    summary = summarize(rows)
    write_csv(RESULT_DIR / "runs.csv", rows)
    write_csv(RESULT_DIR / "summary.csv", summary)
    with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": rows, "summary": summary}, f, indent=2)

    print("\nResumo:")
    for row in summary:
        print(
            f"{row['model']:>5} h={row['hidden']:>3} s={row['states']:>2} "
            f"acc={row['accuracy_mean']*100:>6.2f}% "
            f"flops={row['flops_per_sample']:>8} "
            f"acc/MFLOP={row['accuracy_per_mflop_mean']:.4f}"
        )
    print(f"\nArquivos consolidados em: {RESULT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
