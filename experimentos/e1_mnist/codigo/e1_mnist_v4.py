"""
E1 - MNIST: MLP Tradicional vs V4 Sparse Routing.

Objetivo:
- testar se a V4 mantém accuracy próxima do MLP em MNIST;
- medir FLOPs estimados de inferência;
- medir tempo de treino e inferência;
- registrar uso dos especialistas por classe.

Requisitos de dataset:
Coloque os arquivos IDX do MNIST em datasets/mnist, com estes nomes
(.gz ou descompactados):
- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "datasets" / "mnist"
RESULT_DIR = ROOT / "resultados_finais"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def open_idx(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def find_idx_file(*names: str) -> Path:
    for name in names:
        path = DATA_DIR / name
        if path.exists():
            return path
    expected = " ou ".join(names)
    raise FileNotFoundError(f"Arquivo MNIST nao encontrado em {DATA_DIR}: {expected}")


def read_idx_images(path: Path) -> np.ndarray:
    with open_idx(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Magic invalido para imagens em {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    with open_idx(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Magic invalido para labels em {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n).astype(np.int64)


def load_mnist(train_limit: int | None, test_limit: int | None):
    train_images = find_idx_file("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte")
    train_labels = find_idx_file("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte")
    test_images = find_idx_file("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte")
    test_labels = find_idx_file("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte")

    x_train = read_idx_images(train_images)
    y_train = read_idx_labels(train_labels)
    x_test = read_idx_images(test_images)
    y_test = read_idx_labels(test_labels)

    if train_limit:
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]
    if test_limit:
        x_test = x_test[:test_limit]
        y_test = y_test[:test_limit]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, y_train, x_test, y_test


class LinearLayer:
    def __init__(self, input_dim: int, output_dim: int, seed: int):
        rng = np.random.RandomState(seed)
        self.w = rng.randn(input_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad: np.ndarray, lr: float, l2: float) -> np.ndarray:
        batch = grad.shape[0]
        dw = (self.x.T @ grad) / batch + l2 * self.w
        db = grad.mean(axis=0)
        dx = grad @ self.w.T
        self.w -= lr * dw
        self.b -= lr * db
        return dx

    def params(self) -> int:
        return self.w.size + self.b.size


class DenseMLP:
    def __init__(self, input_dim: int, hidden: int, output_dim: int, seed: int):
        self.l1 = LinearLayer(input_dim, hidden, seed)
        self.l2 = LinearLayer(hidden, hidden, seed + 1)
        self.l3 = LinearLayer(hidden, output_dim, seed + 2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.h1_pre = self.l1.forward(x)
        self.h1 = np.maximum(self.h1_pre, 0)
        self.h2_pre = self.l2.forward(self.h1)
        self.h2 = np.maximum(self.h2_pre, 0)
        return self.l3.forward(self.h2)

    def backward(self, grad: np.ndarray, lr: float, l2: float) -> None:
        grad = self.l3.backward(grad, lr, l2)
        grad[self.h2_pre <= 0] = 0
        grad = self.l2.backward(grad, lr, l2)
        grad[self.h1_pre <= 0] = 0
        self.l1.backward(grad, lr, l2)

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        out = []
        for start in range(0, len(x), batch_size):
            out.append(np.argmax(self.forward(x[start:start + batch_size]), axis=1))
        return np.concatenate(out)

    def params(self) -> int:
        return self.l1.params() + self.l2.params() + self.l3.params()


class SparseV4Layer:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_states: int,
        seed: int,
        gate_hidden: int,
        temperature: float,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_states = n_states
        self.temperature = temperature

        rng = np.random.RandomState(seed)
        rng_gate = np.random.RandomState(seed + 100)
        rng_skip = np.random.RandomState(seed + 200)

        scale = np.sqrt(2.0 / input_dim)
        self.w = [rng.randn(input_dim, output_dim).astype(np.float32) * scale for _ in range(n_states)]
        self.b = [np.zeros(output_dim, dtype=np.float32) for _ in range(n_states)]
        self.gw1 = rng_gate.randn(input_dim, gate_hidden).astype(np.float32) * scale
        self.gb1 = np.zeros(gate_hidden, dtype=np.float32)
        self.gw2 = rng_gate.randn(gate_hidden, n_states).astype(np.float32) * np.sqrt(2.0 / gate_hidden)
        self.gb2 = np.zeros(n_states, dtype=np.float32)
        self.skip_w = rng_skip.randn(input_dim, output_dim).astype(np.float32) * scale
        self.skip_b = np.zeros(output_dim, dtype=np.float32)

    def gate_probs(self, x: np.ndarray) -> np.ndarray:
        h_pre = x @ self.gw1 + self.gb1
        h = np.maximum(h_pre, 0)
        logits = (h @ self.gw2 + self.gb2) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.states = [x @ w + b for w, b in zip(self.w, self.b)]

        self.gate_h_pre = x @ self.gw1 + self.gb1
        self.gate_h = np.maximum(self.gate_h_pre, 0)
        logits = (self.gate_h @ self.gw2 + self.gb2) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        self.probs = exp / exp.sum(axis=1, keepdims=True)

        indices = np.argmax(self.probs, axis=1)
        self.hard = np.zeros_like(self.probs)
        self.hard[np.arange(len(x)), indices] = 1.0

        out = np.zeros((len(x), self.output_dim), dtype=np.float32)
        for i in range(self.n_states):
            out += self.hard[:, i:i + 1] * self.states[i]
        out += x @ self.skip_w + self.skip_b
        return out

    def forward_sparse(self, x: np.ndarray) -> np.ndarray:
        probs = self.gate_probs(x)
        indices = np.argmax(probs, axis=1)
        out = x @ self.skip_w + self.skip_b
        for i in range(self.n_states):
            mask = indices == i
            if np.any(mask):
                out[mask] += x[mask] @ self.w[i] + self.b[i]
        return out

    def backward(self, grad: np.ndarray, lr: float, l2: float) -> np.ndarray:
        batch = grad.shape[0]
        x = self.x

        dskip_w = (x.T @ grad) / batch + l2 * self.skip_w
        dskip_b = grad.mean(axis=0)
        dx_skip = grad @ self.skip_w.T

        dx_states = np.zeros_like(x)
        for i in range(self.n_states):
            g = grad * self.hard[:, i:i + 1]
            dw = (x.T @ g) / batch + l2 * self.w[i]
            db = g.mean(axis=0)
            dx_states += g @ self.w[i].T
            self.w[i] -= lr * dw
            self.b[i] -= lr * db

        d_gate_weight = np.zeros((batch, self.n_states), dtype=np.float32)
        for i in range(self.n_states):
            d_gate_weight[:, i] = np.sum(grad * self.states[i], axis=1)

        sum_term = np.sum(d_gate_weight * self.probs, axis=1, keepdims=True)
        d_logits = self.probs * (d_gate_weight - sum_term)
        d_logits /= self.temperature

        dgw2 = (self.gate_h.T @ d_logits) / batch + l2 * self.gw2
        dgb2 = d_logits.mean(axis=0)
        dgate_h = d_logits @ self.gw2.T
        dgate_h[self.gate_h_pre <= 0] = 0
        dgw1 = (x.T @ dgate_h) / batch + l2 * self.gw1
        dgb1 = dgate_h.mean(axis=0)
        dx_gate = dgate_h @ self.gw1.T

        self.skip_w -= lr * dskip_w
        self.skip_b -= lr * dskip_b
        self.gw2 -= lr * dgw2
        self.gb2 -= lr * dgb2
        self.gw1 -= lr * dgw1
        self.gb1 -= lr * dgb1

        return dx_states + dx_skip + dx_gate

    def params(self) -> int:
        total = sum(w.size for w in self.w) + sum(b.size for b in self.b)
        total += self.gw1.size + self.gb1.size + self.gw2.size + self.gb2.size
        total += self.skip_w.size + self.skip_b.size
        return total


class SparseV4MLP:
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        output_dim: int,
        n_states: int,
        seed: int,
        gate_hidden: int,
        temperature: float,
    ):
        self.l1 = SparseV4Layer(input_dim, hidden, n_states, seed, gate_hidden, temperature)
        self.l2 = SparseV4Layer(hidden, hidden, n_states, seed + 10, gate_hidden, temperature)
        self.l3 = LinearLayer(hidden, output_dim, seed + 20)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.h1_pre = self.l1.forward(x)
        self.h1 = np.maximum(self.h1_pre, 0)
        self.h2_pre = self.l2.forward(self.h1)
        self.h2 = np.maximum(self.h2_pre, 0)
        return self.l3.forward(self.h2)

    def forward_sparse(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(self.l1.forward_sparse(x), 0)
        h2 = np.maximum(self.l2.forward_sparse(h1), 0)
        return h2 @ self.l3.w + self.l3.b

    def backward(self, grad: np.ndarray, lr: float, l2: float) -> None:
        grad = self.l3.backward(grad, lr, l2)
        grad[self.h2_pre <= 0] = 0
        grad = self.l2.backward(grad, lr, l2)
        grad[self.h1_pre <= 0] = 0
        self.l1.backward(grad, lr, l2)

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        out = []
        for start in range(0, len(x), batch_size):
            out.append(np.argmax(self.forward_sparse(x[start:start + batch_size]), axis=1))
        return np.concatenate(out)

    def gate_usage(self, x: np.ndarray, y: np.ndarray, batch_size: int = 512):
        usage_l1 = np.zeros((10, self.l1.n_states), dtype=np.int64)
        usage_l2 = np.zeros((10, self.l2.n_states), dtype=np.int64)
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]
            p1 = self.l1.gate_probs(xb)
            i1 = np.argmax(p1, axis=1)
            h1 = np.maximum(self.l1.forward_sparse(xb), 0)
            p2 = self.l2.gate_probs(h1)
            i2 = np.argmax(p2, axis=1)
            for cls in range(10):
                mask = yb == cls
                if np.any(mask):
                    usage_l1[cls] += np.bincount(i1[mask], minlength=self.l1.n_states)
                    usage_l2[cls] += np.bincount(i2[mask], minlength=self.l2.n_states)
        return usage_l1, usage_l2

    def params(self) -> int:
        return self.l1.params() + self.l2.params() + self.l3.params()


def softmax_crossentropy(logits: np.ndarray, y: np.ndarray):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = y.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-10).mean()
    grad = probs
    grad[np.arange(n), y] -= 1
    return float(loss), grad


def train(model, x_train, y_train, x_test, y_test, epochs, batch_size, lr, l2, sparse_predict):
    history = []
    t0 = time.perf_counter()
    n = len(x_train)
    for epoch in range(1, epochs + 1):
        lr_epoch = lr * (0.97 ** (epoch - 1))
        idx = np.random.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            logits = model.forward(x_train[batch_idx])
            loss, grad = softmax_crossentropy(logits, y_train[batch_idx])
            model.backward(grad, lr_epoch, l2)
            losses.append(loss)
        pred = model.predict(x_test, batch_size) if sparse_predict else model.predict(x_test, batch_size)
        acc = float(np.mean(pred == y_test))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "test_acc": acc})
        print(f"    epoch {epoch:02d}: loss={np.mean(losses):.4f} test_acc={acc*100:.2f}%")
    return history, time.perf_counter() - t0


def time_inference(model, x_test, batch_size, repeats=3):
    best = float("inf")
    last_pred = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_pred = model.predict(x_test, batch_size)
        best = min(best, time.perf_counter() - t0)
    return best, last_pred


def dense_flops(input_dim: int, hidden: int, output_dim: int) -> int:
    return 2 * (input_dim * hidden + hidden * hidden + hidden * output_dim)


def v4_sparse_flops(input_dim: int, hidden: int, output_dim: int, n_states: int, gate_hidden: int) -> int:
    l1_gate = 2 * (input_dim * gate_hidden + gate_hidden * n_states)
    l1_active = 2 * input_dim * hidden
    l1_skip = 2 * input_dim * hidden
    l2_gate = 2 * (hidden * gate_hidden + gate_hidden * n_states)
    l2_active = 2 * hidden * hidden
    l2_skip = 2 * hidden * hidden
    out = 2 * hidden * output_dim
    return l1_gate + l1_active + l1_skip + l2_gate + l2_active + l2_skip + out


def v4_dense_executed_flops(input_dim: int, hidden: int, output_dim: int, n_states: int, gate_hidden: int) -> int:
    l1_gate = 2 * (input_dim * gate_hidden + gate_hidden * n_states)
    l1_states = 2 * n_states * input_dim * hidden
    l1_skip = 2 * input_dim * hidden
    l2_gate = 2 * (hidden * gate_hidden + gate_hidden * n_states)
    l2_states = 2 * n_states * hidden * hidden
    l2_skip = 2 * hidden * hidden
    out = 2 * hidden * output_dim
    return l1_gate + l1_states + l1_skip + l2_gate + l2_states + l2_skip + out


def find_v4_hidden_under_flops(
    target_flops: int,
    input_dim: int,
    output_dim: int,
    n_states: int,
    gate_hidden: int,
    max_hidden: int = 512,
) -> int:
    best = 1
    for hidden in range(1, max_hidden + 1):
        flops = v4_sparse_flops(input_dim, hidden, output_dim, n_states, gate_hidden)
        if flops <= target_flops:
            best = hidden
        else:
            break
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--v4-hidden", type=int, default=0, help="0 escolhe automaticamente por teto de FLOPs")
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--gate-hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--out", default="e1_mnist_v4_result.json")
    args = parser.parse_args()

    set_seed(args.seed)
    train_limit = args.train_limit or None
    test_limit = args.test_limit or None

    print("==================================================")
    print(" E1 MNIST - MLP Tradicional vs V4 Sparse Routing")
    print("==================================================")
    x_train, y_train, x_test, y_test = load_mnist(train_limit, test_limit)
    input_dim = x_train.shape[1]
    output_dim = 10
    print(f"Dataset: train={len(x_train)} test={len(x_test)} input_dim={input_dim}")

    dense_target_flops = dense_flops(input_dim, args.hidden, output_dim)
    if args.v4_hidden <= 0:
        args.v4_hidden = find_v4_hidden_under_flops(
            dense_target_flops,
            input_dim,
            output_dim,
            args.states,
            args.gate_hidden,
        )
        print(f"V4 hidden auto: {args.v4_hidden} (FLOPs_sparse <= baseline)")

    dense = DenseMLP(input_dim, args.hidden, output_dim, args.seed)
    v4 = SparseV4MLP(
        input_dim,
        args.v4_hidden,
        output_dim,
        args.states,
        args.seed,
        args.gate_hidden,
        args.temperature,
    )

    print(f"\nMLP Tradicional: hidden={args.hidden} params={dense.params()}")
    print(f"V4 Sparse: hidden={args.v4_hidden} states={args.states} gate_hidden={args.gate_hidden} params={v4.params()}")

    print("\n--- Treino MLP Tradicional ---")
    dense_history, dense_train_time = train(
        dense, x_train, y_train, x_test, y_test, args.epochs, args.batch_size, args.lr, args.l2, False
    )
    dense_infer_time, dense_pred = time_inference(dense, x_test, args.batch_size)
    dense_acc = float(np.mean(dense_pred == y_test))

    print("\n--- Treino V4 Sparse ---")
    v4_history, v4_train_time = train(
        v4, x_train, y_train, x_test, y_test, args.epochs, args.batch_size, args.lr, args.l2, True
    )
    v4_infer_time, v4_pred = time_inference(v4, x_test, args.batch_size)
    v4_acc = float(np.mean(v4_pred == y_test))

    usage_l1, usage_l2 = v4.gate_usage(x_test, y_test, args.batch_size)

    flops_dense = dense_flops(input_dim, args.hidden, output_dim)
    flops_v4_sparse = v4_sparse_flops(input_dim, args.v4_hidden, output_dim, args.states, args.gate_hidden)
    flops_v4_dense = v4_dense_executed_flops(input_dim, args.v4_hidden, output_dim, args.states, args.gate_hidden)

    result = {
        "experiment": "E1_MNIST",
        "seed": args.seed,
        "dataset": {"train": len(x_train), "test": len(x_test), "input_dim": input_dim, "classes": output_dim},
        "dense": {
            "hidden": args.hidden,
            "params": dense.params(),
            "accuracy": dense_acc,
            "train_time_sec": dense_train_time,
            "inference_time_sec": dense_infer_time,
            "estimated_inference_flops_per_sample": flops_dense,
            "history": dense_history,
        },
        "v4_sparse": {
            "hidden": args.v4_hidden,
            "states": args.states,
            "gate_hidden": args.gate_hidden,
            "params": v4.params(),
            "accuracy": v4_acc,
            "train_time_sec": v4_train_time,
            "inference_time_sec": v4_infer_time,
            "estimated_sparse_inference_flops_per_sample": flops_v4_sparse,
            "estimated_dense_executed_flops_per_sample": flops_v4_dense,
            "history": v4_history,
            "gate_usage_l1_by_class": usage_l1.tolist(),
            "gate_usage_l2_by_class": usage_l2.tolist(),
        },
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n--- Resultado Final ---")
    print(f"MLP Tradicional: acc={dense_acc*100:.2f}% train={dense_train_time:.2f}s infer={dense_infer_time:.4f}s FLOPs={flops_dense}")
    print(
        "V4 Sparse:       "
        f"acc={v4_acc*100:.2f}% train={v4_train_time:.2f}s infer={v4_infer_time:.4f}s "
        f"FLOPs_sparse={flops_v4_sparse} FLOPs_dense_impl={flops_v4_dense}"
    )
    print(f"Resultado salvo em: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"\nERRO: {exc}")
        print("\nColoque os arquivos IDX do MNIST em datasets/mnist e rode novamente.")
        print("Exemplo:")
        print("  python experimentos/e1_mnist/codigo/e1_mnist_v4.py --epochs 5")
        raise SystemExit(2)
