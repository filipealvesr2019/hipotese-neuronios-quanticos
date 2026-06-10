import sys
import os
import numpy as np

# ---------------------------
# Ajusta o path para importar scripts da raiz do projeto
# ---------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"[INFO] Projeto raiz adicionado ao PYTHONPATH: {project_root}")

# ---------------------------
# Importa scripts
# ---------------------------
try:
    from scripts.bateria_completa import LinearLayer, MLPTradicional, make_circles, normalize, train_test_split, set_seed, softmax_crossentropy
except ModuleNotFoundError as e:
    print(f"[ERRO] Não foi possível importar scripts: {e}")
    sys.exit(1)

# ---------------------------
# Importa os experimentos V4 e V5
# ---------------------------
v4_script = os.path.join(os.path.dirname(__file__), "v4_sparse_routing.py")
v5_script = os.path.join(os.path.dirname(__file__), "train.py")

# ---------------------------
# Função para rodar V4
# ---------------------------
def run_v4():
    print("\n======================")
    print("Executando V4 Sparse Routing")
    print("======================\n")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("v4_sparse_routing", v4_script)
        v4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v4)
        print("\n[V4] Execução concluída!\n")
    except Exception as e:
        print(f"[ERRO] Falha ao rodar V4: {e}")

# ---------------------------
# Função para rodar V5
# ---------------------------
def run_v5():
    print("\n======================")
    print("Executando V5 Arena / Competição")
    print("======================\n")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("v5_train", v5_script)
        v5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v5)
        print("\n[V5] Execução concluída!\n")
    except Exception as e:
        print(f"[ERRO] Falha ao rodar V5: {e}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    set_seed(42)
    run_v4()
    run_v5()
    print("[INFO] Arena completa finalizada!")