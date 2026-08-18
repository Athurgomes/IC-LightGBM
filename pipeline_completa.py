import numpy as np
import pandas as pd
from pathlib import Path

# Imports do seu projeto (certifique-se que os caminhos estão corretos)
from src.load_data import carreagar_dados
from src.remover_bandas import remover_bandas_laterais
from src.rois import selecionar_rois
from src.preprocessing import formatar_roi_para_2d
# Pré-processamentos (MSC/SavGol/Z-Score) NÃO são precomputados aqui: são
# aplicados dentro de cada fold na modelagem (anti-leakage). Ver src/modeling_roi.py.

# Import do seu script de construção de dataset
from src.build_dataset import consolidar_datasets_master

CONFIG = {
    "bronze": Path("data/bronze level/hsi_original"),
    "silver": Path("data/silver level"),
    "path_etapa1_calibrados": Path("data/silver level/hsi_calibrado"),
    "path_etapa2_bandas": Path("data/silver level/bandas_removidas"),
    "path_etapa3_rois": Path("data/silver level/rois_extraidos"),
    "path_final_csvs": Path("data/silver level/dados_calibrados"),
    "bandas_remover_inicio": 15,
    "bandas_remover_fim": 15,
    "bandas_rgb_originais": (135, 85, 35),
    "hough_param2": 1
}

def task_calibrarDados(path_in: Path, path_out: Path):
    print("\nETAPA 1: LER E CALIBRAR OS DADOS")
    path_out.mkdir(parents=True, exist_ok=True)
    for sample_dir in path_in.iterdir():
        if not sample_dir.is_dir(): continue
        sample_name = sample_dir.name
        print(f"Processando: {sample_name}")
        path_capture = sample_dir / "capture"
        path_raw = path_capture / f"{sample_name}.hdr"
        path_white = path_capture / f"WHITEREF_{sample_name}.hdr"
        path_dark = path_capture / f"DARKREF_{sample_name}.hdr"
        if not all([p.exists() for p in [path_raw, path_white, path_dark]]):
            print(f"  AVISO: Arquivos faltando para {sample_name}. Pulando.")
            continue
        calibrated_cube, _ = carreagar_dados(str(path_raw), str(path_white), str(path_dark))
        if calibrated_cube is not None:
            output_path = path_out / f"{sample_name}.npy"
            np.save(output_path, calibrated_cube)
            print(f"  SUCESSO: Salvo em {output_path.name}")
    print("ETAPA 1 CONCLUÍDA")

def task_removerBandas(path_in: Path, path_out: Path, inicio: int, fim: int):
    print("\nETAPA 2: REMOÇÃO DE BANDAS")
    path_out.mkdir(parents=True, exist_ok=True)
    for npy_file in path_in.glob("*.npy"):
        cubo = np.load(npy_file)
        cubo_cortado = remover_bandas_laterais(cubo, bandas_inicio=inicio, bandas_fim=fim)
        if cubo_cortado is not None:
            output_path = path_out / npy_file.name
            np.save(output_path, cubo_cortado)
            print(f"  SUCESSO: {npy_file.name} cortado.")
    print("ETAPA 2 CONCLUÍDA")

def task_extrairRois(path_in: Path, path_out: Path, bandas_rgb: tuple):
    print("\nETAPA 3: EXTRAÇÃO DE ROIS")
    path_out.mkdir(parents=True, exist_ok=True)
    bandas_ajustadas = (
        bandas_rgb[0] - CONFIG["bandas_remover_inicio"],
        bandas_rgb[1] - CONFIG["bandas_remover_inicio"],
        bandas_rgb[2] - CONFIG["bandas_remover_inicio"]
    )
    for npy_file in path_in.glob("*.npy"):
        cubo = np.load(npy_file)
        roi_extraido, _, _, mask_2d = selecionar_rois(cubo, bandas=bandas_ajustadas, param2_ajustado=CONFIG["hough_param2"])
        if roi_extraido is not None:
            output_path = path_out / f"{npy_file.stem}.npz"
            np.savez(output_path, cube=roi_extraido, mask=mask_2d)
            print(f"  SUCESSO: ROI extraído de {npy_file.name}")
    print("ETAPA 3 CONCLUÍDA")

def _carregar_rois(path_in: Path):
    dados = {}
    for npz_file in sorted(path_in.glob("*.npz")):
        pacote = np.load(npz_file)
        X_roi = formatar_roi_para_2d(pacote["cube"], mask_2d=pacote["mask"])
        if X_roi is not None: dados[npz_file.stem] = X_roi
    return dados

def task_gerarSilverRaw(path_in: Path, path_base_out: Path):
    """Salva o RAW calibrado (espectros da ROI) por amostra — silver = só RAW.

    Antes este passo precomputava MSC/SavGol/Z-Score com estatísticas GLOBAIS
    (referência e scaler sobre todas as amostras) → vazamento se usado em
    treino/teste. Agora o silver guarda só o RAW; os pré-processamentos são
    ajustados dentro de cada fold na modelagem (src/modeling_roi.py). Os
    notebooks de exploração (04/05) computam as variantes on-the-fly do RAW.
    """
    print("\nETAPA 4: GERAÇÃO DO SILVER RAW (calibrado) POR AMOSTRA")
    dados_amostras = _carregar_rois(path_in)
    if not dados_amostras: return

    pasta_raw = path_base_out / "raw"
    pasta_raw.mkdir(parents=True, exist_ok=True)

    for sample_name, X_roi in dados_amostras.items():
        try:
            df = pd.DataFrame(X_roi)
            df.to_csv(pasta_raw / f"{sample_name}.csv", index=False, header=False)
        except Exception as e:
            print(f"  ERRO em {sample_name}: {e}")
    print("ETAPA 4 CONCLUÍDA")

def main():
    print("=" * 40)
    print("INICIANDO PIPELINE DE PROCESSAMENTO HSI")
    print("=" * 40)
    
    task_calibrarDados(CONFIG["bronze"], CONFIG["path_etapa1_calibrados"])
    task_removerBandas(CONFIG["path_etapa1_calibrados"], CONFIG["path_etapa2_bandas"], 15, 15)
    task_extrairRois(CONFIG["path_etapa2_bandas"], CONFIG["path_etapa3_rois"], CONFIG["bandas_rgb_originais"])
    task_gerarSilverRaw(CONFIG["path_etapa3_rois"], CONFIG["path_final_csvs"])
    
    # Chama o build_dataset para consolidar os arquivos e criar os labels
    print("\nETAPA 5: CONSOLIDAÇÃO FINAL (BUILD DATASET)")
    consolidar_datasets_master(CONFIG["path_final_csvs"])
    
    print("\nPIPELINE COMPLETO EXECUTADO COM SUCESSO!")

if __name__ == "__main__":
    main()