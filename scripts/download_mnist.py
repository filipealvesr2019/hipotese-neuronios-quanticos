"""
Baixa os arquivos IDX do MNIST para datasets/mnist e valida SHA-256.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "mnist"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"

FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, expected_hash: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name

    if path.exists():
        current_hash = sha256(path)
        if current_hash == expected_hash:
            print(f"OK   {name}")
            return
        print(f"Hash invalido, baixando novamente: {name}")

    url = f"{BASE_URL}/{name}"
    print(f"GET  {url}")
    urllib.request.urlretrieve(url, path)

    current_hash = sha256(path)
    if current_hash != expected_hash:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 invalido para {name}: {current_hash}")
    print(f"OK   {name}")


def main() -> int:
    for name, expected_hash in FILES.items():
        download(name, expected_hash)
    print(f"\nMNIST pronto em: {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
