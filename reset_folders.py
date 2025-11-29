import shutil
from pathlib import Path
import sys

BASE_SILVER = Path("data/silver level")
PASTAS_PARA_LIMPAR = [
    BASE_SILVER / "hsi_calibrado",
    BASE_SILVER / "bandas_removidas",
    BASE_SILVER / "rois_extraidos",
    BASE_SILVER / "dados_calibrados"
]

def limpar_tudo():
    print("="*40)
    print("MODO DE LIMPEZA DE DADOS")
    print("="*40)
    print("As seguintes pastas (e todo o seu conteúdo) serão apagadas:\n")
    for pasta in PASTAS_PARA_LIMPAR:
        status = "Existe" if pasta.exists() else "Não existe (será ignorada)"
        print(f" - {pasta} [{status}]")
    confirmacao = input("\nTem certeza que deseja DELETAR todos esses dados? (digite 'sim'): ")
    if confirmacao.lower() != 'sim':
        print("\nOperação cancelada. Nenhum dado foi apagado.")
        sys.exit()
    print("\nIniciando limpeza...")
    for pasta in PASTAS_PARA_LIMPAR:
        if pasta.exists():
            try:
                shutil.rmtree(pasta)
                print(f"Apagado: {pasta}")
            except Exception as e:
                print(f"Erro ao apagar {pasta}: {e}")
        else:
            print(f"Pulado (não existia): {pasta}")
    print("\nLIMPEZA CONCLUÍDA!")

if __name__ == "__main__":
    limpar_tudo()