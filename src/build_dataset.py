import pandas as pd
from pathlib import Path


def _carregar_mapa_rotulos(metadata_path: Path) -> dict:
    """Lê metadata.csv (fonte de verdade) e devolve {sample_id: target}.

    Substitui a regra frágil `1 if "ATCC" in nome` — o rótulo agora vem de um
    arquivo auditável (gerado por src/build_metadata.py e revisado contra os
    exames microbiológicos).
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.csv não encontrado em {metadata_path}. "
            "Gere com: python src/build_metadata.py"
        )
    meta = pd.read_csv(metadata_path)
    return dict(zip(meta["sample_id"], meta["target"].astype(int)))


def consolidar_datasets_master(caminho_base: Path, metadata_path: Path = None):
    """
    Consolida os CSVs RAW (calibrados) gerados pelo pipeline em master_raw.csv.

    Silver guarda APENAS o RAW calibrado — os pré-processamentos (MSC, SavGol,
    Z-Score) são aplicados DENTRO de cada fold na modelagem (anti-leakage),
    não precomputados globalmente. O rótulo vem do metadata.csv.
    """
    caminho_base = Path(caminho_base)
    if metadata_path is None:
        # data/silver level/dados_calibrados → data/metadata.csv
        metadata_path = caminho_base.resolve().parents[1] / "metadata.csv"
    mapa_rotulos = _carregar_mapa_rotulos(metadata_path)

    metodos = ["raw"]  # silver = só RAW; pré-proc é feito no fold

    for metodo in metodos:
        pasta_metodo = caminho_base / metodo

        if not pasta_metodo.exists():
            print(f"[AVISO] Pasta '{metodo}' não encontrada. Pulando.")
            continue

        arquivos_csv = sorted(pasta_metodo.glob("*.csv"))

        if not arquivos_csv:
            print(f"[AVISO] Nenhum CSV encontrado em '{metodo}'. Pulando.")
            continue

        print(f"\n[INFO] Consolidando método: {metodo}")
        lista_dfs = []

        for csv_path in arquivos_csv:
            # Carrega o CSV (etapa de pré-processamento salva sem header)
            df = pd.read_csv(csv_path, header=None)

            # Rótulo pelo metadata.csv (não pelo nome do arquivo)
            sample_id = csv_path.stem
            if sample_id not in mapa_rotulos:
                raise KeyError(
                    f"sample_id '{sample_id}' ausente no metadata.csv "
                    f"({metadata_path}). Atualize o metadata."
                )
            label = mapa_rotulos[sample_id]

            # Adiciona colunas de identificação e target
            df.insert(0, "sample_id", sample_id)
            df["target"] = label

            lista_dfs.append(df)

        # Concatenação e nomeação de colunas
        df_master = pd.concat(lista_dfs, ignore_index=True)

        # Nomeia colunas: sample_id, b0, b1... target
        n_bandas = df_master.shape[1] - 2
        colunas = ["sample_id"] + [f"b{i}" for i in range(n_bandas)] + ["target"]
        df_master.columns = colunas

        # Salvamento
        output_path = caminho_base / f"master_{metodo}.csv"
        df_master.to_csv(output_path, index=False)

        print(f"[OK] {output_path.name} salvo. Shape: {df_master.shape}")
        print(df_master["target"].value_counts())


if __name__ == "__main__":
    # Caminho para a pasta silver level/dados_calibrados
    ROOT_DIR = Path(__file__).resolve().parents[1]
    DIRETORIO_SILVER = ROOT_DIR / "data" / "silver level" / "dados_calibrados"
    consolidar_datasets_master(DIRETORIO_SILVER)
