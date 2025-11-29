import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil 

from src.load_data import carreagar_dados
from src.remover_bandas import remover_bandas_laterais
from src.rois import selecionar_rois
from src.preprocessing import (
    formatar_roi_para_2d, 
    apply_msc, 
    apply_savitzky_golay, 
    apply_zscore
)

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
        else:
            print(f"  ERRO: Falha ao calibrar {sample_name}.")
    print("ETAPA 1 CONCLUÍDA")

def task_removerBandas(path_in: Path, path_out: Path, inicio: int, fim: int):
    print("\nETAPA 2: REMOÇÃO DE BANDAS")
    path_out.mkdir(parents=True, exist_ok=True)
    for npy_file in path_in.glob("*.npy"):
        print(f"Processando: {npy_file.name}")
        cubo = np.load(npy_file)  
        cubo_cortado = remover_bandas_laterais(cubo, bandas_inicio=inicio, bandas_fim=fim)  
        if cubo_cortado is not None:
            output_path = path_out / npy_file.name
            np.save(output_path, cubo_cortado)
            print(f"  SUCESSO: Salvo com {cubo_cortado.shape[2]} bandas.")
    print("ETAPA 2 CONCLUÍDA")

def task_extrairRois(path_in: Path, path_out: Path, bandas_rgb: tuple):
    print("\nETAPA 3: EXTRAÇÃO DE ROIS")
    path_out.mkdir(parents=True, exist_ok=True)
    bandas_ajustadas = (
        bandas_rgb[0] - CONFIG["bandas_remover_inicio"],
        bandas_rgb[1] - CONFIG["bandas_remover_inicio"],
        bandas_rgb[2] - CONFIG["bandas_remover_inicio"]
    )
    print(f"(Usando bandas {bandas_ajustadas} para detecção de ROI)")
    for npy_file in path_in.glob("*.npy"):
        print(f"Processando: {npy_file.name}")
        cubo = np.load(npy_file) 
        roi_extraido, _, _ = selecionar_rois(cubo, bandas=bandas_ajustadas, param2_ajustado=CONFIG["hough_param2"])
        if roi_extraido is not None:
            output_path = path_out / npy_file.name
            np.save(output_path, roi_extraido)
            print(f"  SUCESSO: ROI extraído e salvo.")
        else:
            print(f"  AVISO: Nenhum ROI detectado para {npy_file.name}. Pulando.")
    print("ETAPA 3 CONCLUÍDA")
    
def task_preProcessamentoIndependente(path_in: Path, path_base_out: Path):
    print("\nETAPA 4: PRE-PROCESSAMENTO (MÉTODOS INDEPENDENTES)")
    metodos = {
        "raw": lambda x: x,
        "msc": apply_msc,
        "savigol": apply_savitzky_golay,
        "z_score": apply_zscore
    } 
    for pasta in metodos.keys():
        (path_base_out / pasta).mkdir(parents=True, exist_ok=True)
    for npy_file in path_in.glob("*.npy"):
        sample_name = npy_file.stem
        print(f"Gerando CSVs para: {sample_name}")       
        cubo_roi = np.load(npy_file)
        X_roi = formatar_roi_para_2d(cubo_roi)      
        if X_roi is None:
            print(f"  AVISO: ROI vazio. Pulando.")
            continue           
        for nome_metodo, funcao in metodos.items():
            try:
                X_final = funcao(X_roi)            
                df = pd.DataFrame(X_final)
                output_path = path_base_out / nome_metodo / f"{sample_name}.csv"
                df.to_csv(output_path, index=False, header=False)         
            except Exception as e:
                print(f"  ERRO no método {nome_metodo}: {e}")           
    print("ETAPA 4 CONCLUÍDA")

def main():
    print("="*40)
    print("INICIANDO PIPELINE DE PROCESSAMENTO HSI")
    print("="*40)
    task_calibrarDados(CONFIG["bronze"], CONFIG["path_etapa1_calibrados"])
    task_removerBandas(CONFIG["path_etapa1_calibrados"], CONFIG["path_etapa2_bandas"], CONFIG["bandas_remover_inicio"], CONFIG["bandas_remover_fim"])
    task_extrairRois(CONFIG["path_etapa2_bandas"], CONFIG["path_etapa3_rois"], CONFIG["bandas_rgb_originais"])
    task_preProcessamentoIndependente(CONFIG["path_etapa3_rois"], CONFIG["path_final_csvs"])
    print("\nPIPELINE CONCLUÍDO COM SUCESSO!")

if __name__ == "__main__":
    main()