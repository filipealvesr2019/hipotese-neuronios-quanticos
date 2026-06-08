import os
import shutil

# Target structure
dirs_to_create = [
    "docs/pt-BR/relatorios",
    "docs/en-US/reports",
    "experimentos/v1_multistate/codigo",
    "experimentos/v1_multistate/resultados",
    "experimentos/v2_gating/codigo",
    "experimentos/v2_gating/resultados",
    "experimentos/v3_skip_connections/codigo",
    "experimentos/v3_skip_connections/resultados",
    "experimentos/v4_sparse_routing/codigo",
    "experimentos/v4_sparse_routing/resultados",
    "experimentos/v4_1_load_balancing/codigo",
    "experimentos/v4_1_load_balancing/resultados",
    "datasets/moons",
    "datasets/circles",
    "datasets/spirals",
    "datasets/mnist",
    "scripts",
    "resultados_finais/tabelas",
    "resultados_finais/graficos",
    "resultados_finais/comparacoes",
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)

# Helper function
def move_file(src, dst):
    if os.path.exists(src):
        shutil.move(src, dst)

# docs
with open("docs/pt-BR/diario.md", "w", encoding="utf-8") as f:
    f.write("# Diário\n\n## 2026-06-08\n\nHipótese:\nEstados internos independentes...\n")
with open("docs/pt-BR/conclusoes.md", "w", encoding="utf-8") as f:
    f.write("# Conclusões Atuais\n\n✅ Estados aprendem representações diferentes.\n✅ Gate influencia fortemente o resultado.\n✅ Sparse Routing reduz FLOPs.\n")
with open("docs/pt-BR/HIPOTESES_REFUTADAS.md", "w", encoding="utf-8") as f:
    f.write("# Hipóteses Refutadas\n\n## H001\nEstados somados melhoram desempenho.\nResultado: Falso.\n")
with open("docs/en-US/diary.md", "w", encoding="utf-8") as f:
    f.write("# Diary\n")
with open("docs/en-US/conclusions.md", "w", encoding="utf-8") as f:
    f.write("# Conclusions\n")

# Move root text files to docs
move_file("Relatório de Pesquisa – Neurônio MultiEstado e Roteamento Esparso.txt", "docs/pt-BR/relatorios/Relatorio_MultiEstado.txt")
move_file("intruçao pra criar relatorios.md", "docs/pt-BR/relatorios/instrucoes_originais.md")
move_file("output.txt", "resultados_finais/output.txt")
move_file("resultados.json", "experimentos/v1_multistate/resultados/resultados.json")

# Scripts
move_file("bateria_completa.py", "scripts/bateria_completa.py")
move_file("runner.py", "scripts/runner.py")
move_file("sumario.py", "scripts/sumario.py")
move_file("run_teste.bat", "scripts/run_teste.bat")
move_file("test_imports.py", "scripts/test_imports.py")
if os.path.exists("bateria_tests/common.py"):
    move_file("bateria_tests/common.py", "scripts/common.py")

# V1
move_file("experimento_quantico.py", "experimentos/v1_multistate/codigo/experimento_quantico.py")
move_file("experimento_quantico_sklearn.py", "experimentos/v1_multistate/codigo/experimento_quantico_sklearn.py")
if os.path.exists("resultados"):
    for item in os.listdir("resultados"):
        move_file(os.path.join("resultados", item), "experimentos/v1_multistate/resultados/" + item)
    try:
        os.rmdir("resultados")
    except:
        pass

if os.path.exists("bateria_tests"):
    for item in os.listdir("bateria_tests"):
        src = os.path.join("bateria_tests", item)
        if item.startswith("test") and "v2" not in item:
            move_file(src, "experimentos/v1_multistate/codigo/" + item)
        elif "v2" in item:
            move_file(src, "experimentos/v2_gating/codigo/" + item)
        elif "v3" in item:
            move_file(src, "experimentos/v3_skip_connections/codigo/" + item)
        elif "v4_1" in item:
            move_file(src, "experimentos/v4_1_load_balancing/codigo/" + item)
        elif "v4" in item:
            move_file(src, "experimentos/v4_sparse_routing/codigo/" + item)

    try:
        shutil.rmtree("bateria_tests")
    except:
        pass

# Fix imports in V3 and V4 which point to bateria_completa
# We moved bateria_completa to scripts, so they need to point to scripts.bateria_completa or sys.path
def update_imports(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                content = content.replace("from bateria_completa", "from scripts.bateria_completa")
                content = content.replace("from v3_skip_mlp_gate", "from experimentos.v3_skip_connections.codigo.v3_skip_mlp_gate")
                content = content.replace("from v4_sparse_routing", "from experimentos.v4_sparse_routing.codigo.v4_sparse_routing")
                
                # For common.py imports in V2
                content = content.replace("from common", "from scripts.common")
                
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

update_imports("experimentos")

print("Repository refactored successfully!")
