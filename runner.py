"""
Runner sequencial: executa cada teste individualmente,
salva resultados em JSON, prossegue mesmo se um falhar.
"""
import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(BASE, "bateria_tests")

# Tests em ordem progressiva de complexidade
TESTS = [
    ("test1_XOR.py",             "XOR"),
    ("test2_paridade.py",        "Paridade"),
    ("test3_fronteiras.py",      "Fronteiras nao lineares"),
    ("test4_compressao.py",      "Compressao"),
    ("test5_capacidade.py",      "Capacidade"),
    ("test6_poucos_dados.py",    "Poucos dados"),
    ("test7_ruido.py",           "Ruido"),
    ("test8_especializacao.py",  "Especializacao de estados"),
    ("test9_ablacao.py",         "Ablacao (n_states)"),
    ("test10_info_por_param.py", "Info por parametro"),
    ("test11_destilacao.py",     "Destilacao"),
    ("test12_correlacao.py",     "Correlacao entre estados"),
]

def main():
    print("=" * 60)
    print("BATERIA COMPLETA DE TESTES - RUNNER SEQUENCIAL")
    print("=" * 60)
    print(f"  Diretorio: {TESTS_DIR}")
    print(f"  Total de testes: {len(TESTS)}")
    print()

    passed = 0
    failed = 0

    for i, (filename, description) in enumerate(TESTS, 1):
        filepath = os.path.join(TESTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[{i}/{len(TESTS)}] {description} -> ARQUIVO NAO ENCONTRADO: {filepath}")
            failed += 1
            continue

        print(f"\n>>> EXECUTANDO TESTE {i}/{len(TESTS)}: {description} ({filename})")
        print(f"{'='*60}")
        sys.stdout.flush()

        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=BASE
        )

        if result.returncode == 0:
            passed += 1
            print(result.stdout)
        else:
            failed += 1
            print(f"  *** FALHOU (codigo {result.returncode}) ***")
            if result.stdout:
                print(result.stdout[:2000])
            if result.stderr:
                print(f"  STDERR: {result.stderr[:1000]}")

        sys.stdout.flush()

    # Final report
    print("\n" + "=" * 60)
    print(f"RESUMO: {passed}/{len(TESTS)} testes passaram, {failed} falharam")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
