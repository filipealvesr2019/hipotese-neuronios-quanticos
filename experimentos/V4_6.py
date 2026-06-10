import numpy as np
import json
import os


# =========================
# V4.6 — MoE Credit Assignment FIXED
# =========================
class V46MoE:
    def __init__(self, input_dim, hidden, output_dim,
                 n_states=2, seed=42, lr=0.01,
                 temperature=1.2, entropy_reg=0.002):

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.n_states = n_states
        self.lr = lr
        self.temperature = temperature
        self.entropy_reg = entropy_reg

        rng = np.random.RandomState(seed)

        # =====================
        # Experts
        # =====================
        self.W1 = [rng.randn(input_dim, hidden) * 0.1 for _ in range(n_states)]
        self.b1 = [np.zeros(hidden) for _ in range(n_states)]
        self.W2 = [rng.randn(hidden, output_dim) * 0.1 for _ in range(n_states)]
        self.b2 = [np.zeros(output_dim) for _ in range(n_states)]

        # =====================
        # Gate
        # =====================
        self.gW1 = rng.randn(input_dim, 16) * 0.1
        self.gb1 = np.zeros(16)
        self.gW2 = rng.randn(16, n_states) * 0.1
        self.gb2 = np.zeros(n_states)

    # =====================
    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def softmax_1d(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    # =====================
    # FORWARD (FIXED)
    # =====================
    def forward(self, x, add_noise=False):
        B = x.shape[0]

        h_states = []
        out_states = []

        for i in range(self.n_states):
            h = self.relu(x @ self.W1[i] + self.b1[i])
            out = h @ self.W2[i] + self.b2[i]
            h_states.append(h)
            out_states.append(out)

        g = self.relu(x @ self.gW1 + self.gb1)
        logits_g = g @ self.gW2 + self.gb2

        if add_noise:
            logits_g += np.random.normal(0, 0.05, logits_g.shape)

        gate = self.softmax(logits_g / self.temperature)

        out = sum(gate[:, i:i+1] * out_states[i] for i in range(self.n_states))

        return out, gate, out_states, h_states, g

    # =====================
    def loss(self, logits, y):
        probs = self.softmax(logits)
        return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9))

    # =====================
    # TRAIN STEP (FIXED CORE)
    # =====================
    def train_step(self, x, y):
        B = x.shape[0]

        logits, gate, out_states, h_states, g = self.forward(x, add_noise=True)

        # =====================
        # OUTPUT GRADIENT
        # =====================
        probs = self.softmax(logits)
        probs[np.arange(B), y] -= 1
        probs /= B

        # =====================
        # EXPERT UPDATES
        # =====================
        expert_loss = np.zeros((B, self.n_states))

        for i in range(self.n_states):

            grad = probs * gate[:, i:i+1]

            dh = grad @ self.W2[i].T
            dh[h_states[i] <= 0] = 0

            self.W2[i] -= self.lr * (h_states[i].T @ grad)
            self.b2[i] -= self.lr * np.sum(grad, axis=0)
            self.W1[i] -= self.lr * (x.T @ dh)
            self.b1[i] -= self.lr * np.sum(dh, axis=0)

            # proxy loss por expert
            pi = self.softmax(out_states[i])
            expert_loss[:, i] = -np.log(pi[np.arange(B), y] + 1e-9)

        # =====================
        # CREDIT ASSIGNMENT (FIXED)
        # =====================
        reward = -expert_loss

        reward = reward - reward.mean(axis=1, keepdims=True)
        reward = reward / (np.std(reward) + 1e-9)

        gate_grad = reward - gate

        # =====================
        # GATE BACKPROP
        # =====================
        dg = gate_grad @ self.gW2.T
        dg[g <= 0] = 0

        self.gW2 += self.lr * (g.T @ gate_grad)
        self.gb2 += self.lr * np.sum(gate_grad, axis=0)
        self.gW1 += self.lr * (x.T @ dg)
        self.gb1 += self.lr * np.sum(dg, axis=0)

        # =====================
        # entropy regularization
        # =====================
        ent = -np.sum(gate.mean(axis=0) * np.log(gate.mean(axis=0) + 1e-9))

        return self.loss(logits, y) - self.entropy_reg * ent


# =====================
# DATASET
# =====================
def make_data(n=2000, d=784, c=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return X, y


def accuracy(model, X, y):
    logits, _, _, _, _ = model.forward(X)
    return np.mean(np.argmax(logits, axis=1) == y)


def entropy(p):
    p = p + 1e-9
    return -np.sum(p * np.log(p))


# =====================
# RUN EXPERIMENT
# =====================
def run():
    results = []

    for seed in range(10):
        print(f"\n=== Seed {seed} ===")

        X, y = make_data(seed=seed)

        model = V46MoE(
            input_dim=784,
            hidden=96,
            output_dim=10,
            n_states=2,
            seed=seed
        )

        for epoch in range(3):
            idx = np.random.permutation(len(X))
            for i in range(0, len(X), 64):
                batch = idx[i:i+64]
                model.train_step(X[batch], y[batch])

        acc = accuracy(model, X, y)

        _, gate, _, _, _ = model.forward(X)
        ent = entropy(np.mean(gate, axis=0))

        print(f"Acc={acc:.4f} | Entropy={ent:.4f}")

        results.append({
            "seed": seed,
            "acc": float(acc),
            "entropy": float(ent)
        })

    os.makedirs("resultados_finais", exist_ok=True)

    with open("resultados_finais/v4_6.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved -> resultados_finais/v4_6.json")


if __name__ == "__main__":
    run()