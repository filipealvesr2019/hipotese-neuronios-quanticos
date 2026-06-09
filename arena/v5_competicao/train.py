# train.py - V5 Competicao
import numpy as np
import random
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ---------------------------
# Função de treino principal
# ---------------------------
def train(epochs=5, batch_size=128, hidden=96, states=2, seed=1, log_gate=False):
    """
    Treino simplificado V5 - Especialistas sem gate externo
    """

    np.random.seed(seed)
    random.seed(seed)

    # ---------------------------
    # Dataset sintético MNIST-like (784 features)
    # ---------------------------
    n_features = 784
    n_classes = 10
    X, y = make_classification(n_samples=5000, n_features=n_features, n_informative=50,
                               n_classes=n_classes, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # ---------------------------
    # Inicializar especialistas
    # states especialistas com pesos aleatórios
    # ---------------------------
    experts = []
    for s in range(states):
        W1 = np.random.randn(n_features, hidden) * 0.01
        b1 = np.zeros(hidden)
        W2 = np.random.randn(hidden, n_classes) * 0.01
        b2 = np.zeros(n_classes)
        experts.append({"W1": W1, "b1": b1, "W2": W2, "b2": b2})

    # ---------------------------
    # Funções auxiliares
    # ---------------------------
    def softmax(z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / e_z.sum(axis=1, keepdims=True)

    def forward(x, expert):
        h = np.maximum(0, x.dot(expert["W1"]) + expert["b1"])  # ReLU
        out = h.dot(expert["W2"]) + expert["b2"]
        return out

    def accuracy(y_true, y_pred):
        return (y_true == y_pred).mean() * 100

    # ---------------------------
    # Treino
    # ---------------------------
    n_train = X_train.shape[0]
    result_log = {"epochs": []}

    for epoch in range(epochs):
        indices = np.arange(n_train)
        np.random.shuffle(indices)
        batch_losses = []
        batch_accs = []

        for start in range(0, n_train, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # Escolher especialista aleatório (simples Top-1 pseudo-random)
            chosen = np.random.randint(states)
            expert = experts[chosen]

            logits = forward(X_batch, expert)
            probs = softmax(logits)
            preds = np.argmax(probs, axis=1)
            acc = accuracy(y_batch, preds)

            batch_accs.append(acc)
            # Loss cross-entropy simplificada
            eps = 1e-12
            correct_logprobs = -np.log(probs[np.arange(len(y_batch)), y_batch] + eps)
            batch_losses.append(np.mean(correct_logprobs))

            # ---------------------------
            # Gradient descent (simplificado)
            # ---------------------------
            grad_out = probs
            grad_out[np.arange(len(y_batch)), y_batch] -= 1
            grad_out /= len(y_batch)

            h = np.maximum(0, X_batch.dot(expert["W1"]) + expert["b1"])
            grad_W2 = h.T.dot(grad_out)
            grad_b2 = grad_out.sum(axis=0)

            dh = grad_out.dot(expert["W2"].T)
            dh[h <= 0] = 0
            grad_W1 = X_batch.T.dot(dh)
            grad_b1 = dh.sum(axis=0)

            lr = 0.01
            expert["W1"] -= lr * grad_W1
            expert["b1"] -= lr * grad_b1
            expert["W2"] -= lr * grad_W2
            expert["b2"] -= lr * grad_b2

        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_accs)
        result_log["epochs"].append({"loss": epoch_loss, "accuracy": epoch_acc})
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Expert: {chosen}")

    # Avaliação final
    all_preds = []
    for x in X_test:
        chosen = np.random.randint(states)
        expert = experts[chosen]
        logits = forward(x.reshape(1,-1), expert)
        pred = np.argmax(logits)
        all_preds.append(pred)
    test_acc = accuracy(y_test, np.array(all_preds))

    print(f"Treino concluído! Test Accuracy: {test_acc:.2f}%")
    return {"test_accuracy": test_acc, "epochs_log": result_log, "states": states, "hidden": hidden}
