"""
feature_engineering.py — Tópico 3: Engenharia de Atributos

Abordagem: PIXEL COMO AMOSTRA
══════════════════════════════════════════════════════════════════════════════
Cada pixel é tratado como uma amostra independente, aumentando o dataset
de 19 linhas (amostras biológicas) para N_pixels linhas.

Para cada pixel, são extraídas as features do plano de trabalho:

  ┌─ Intensidade das k bandas selecionadas  → band_0 .. band_{k-1}
  ├─ Estatísticas espectrais do pixel        → mean, std, skewness, kurtosis
  │    (calculadas sobre os k valores de banda daquele pixel)
  ├─ Derivadas locais entre bandas adjacentes → deriv_0 .. deriv_{k-2}
  └─ Índices espectrais personalizados        → SR_max_min, NDSI_edge, total_slope

Com k=15 bandas e incluir_features_derivadas=True:
  15 + 4 + 14 + 3 = 36 features por pixel

Com k=15 bandas e incluir_features_derivadas=False (somente espectros):
  15 features por pixel

Seleção de bandas (MI / ANOVA / LGBM FI) → feature_selectors.py
Avaliação no nível de amostra            → modeling_roi.py
══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pathlib import Path


COLUNAS_META = ['sample_id', 'target', 'block_id', 'roi_id', 'pixel_id']


def build_gold_matrix(
    df_silver,
    bandas_selecionadas=None,
    incluir_features_derivadas=True,
):
    """
    Constrói a matriz Gold no nível de PIXEL.

    Parâmetros
    ----------
    df_silver                : DataFrame Silver (1 linha por pixel).
    bandas_selecionadas      : lista de colunas de bandas a usar, ou None
                               (None → usa TODAS as bandas — Baseline).
    incluir_features_derivadas : bool (padrão True).
        True  → pipeline completo: bandas + stats + derivadas + índices.
        False → somente intensidades das k bandas (sem nenhuma feature derivada).
                Usado para comparar o ganho real das features derivadas.

    Retorna
    -------
    DataFrame Gold com N_pixels linhas.

    Tamanhos com incluir_features_derivadas=True:
      k=15  → 15 + 4 + 14 + 3 = 36 features por pixel
      k=226 → 226 + 4 + 225 + 3 = 458 features (baseline completo)

    Tamanhos com incluir_features_derivadas=False:
      k=15  → 15 features por pixel (somente espectros)
      k=226 → 226 features por pixel (baseline puro)
    """
    todas_band_cols = [c for c in df_silver.columns if c not in COLUNAS_META]
    band_cols = bandas_selecionadas if bandas_selecionadas is not None else todas_band_cols

    meta = [c for c in ['sample_id', 'target'] if c in df_silver.columns]
    valores      = df_silver[band_cols].values        # (N_pixels, k) — bandas selecionadas
    valores_full = df_silver[todas_band_cols].values  # (N_pixels, 226) — espectro completo
    # stats e índices usam o espectro completo para preservar significado físico;
    # quando bandas_selecionadas is None, valores_full == valores (sem diferença).

    # ── 1. Intensidades das k bandas selecionadas ─────────────────────────────
    df_bandas = pd.DataFrame(
        valores,
        columns=[f'band_{i}' for i in range(len(band_cols))],
    )

    blocos = [df_silver[meta].reset_index(drop=True), df_bandas]

    if incluir_features_derivadas:
        # ── 2. Estatísticas espectrais por pixel (sobre espectro completo) ─────
        df_stats = pd.DataFrame({
            'mean':     valores_full.mean(axis=1),
            'std':      valores_full.std(axis=1),
            'skewness': [float(skew(row))     for row in valores_full],
            'kurtosis': [float(kurtosis(row)) for row in valores_full],
        })

        # ── 3. Derivadas locais entre bandas adjacentes ───────────────────────
        derivadas = np.diff(valores, axis=1)
        df_derivadas = pd.DataFrame(
            derivadas,
            columns=[f'deriv_{i}' for i in range(derivadas.shape[1])],
        )

        # ── 4. Índices espectrais personalizados (sobre espectro completo) ────
        max_v  = valores_full.max(axis=1)
        min_v  = valores_full.min(axis=1)
        first  = valores_full[:, 0]
        last   = valores_full[:, -1]
        k      = valores_full.shape[1]

        df_indices = pd.DataFrame({
            'SR_max_min':  max_v / (min_v + 1e-6),
            'NDSI_edge':   (last - first) / (last + first + 1e-6),
            'total_slope': (last - first) / k,
        })

        blocos += [df_stats, df_derivadas, df_indices]

    df_gold = pd.concat(blocos, axis=1)
    return df_gold


# =============================================================================
# PERSISTÊNCIA — SALVAR GOLD COMO CSV
# =============================================================================

def salvar_gold(
    df_silver,
    bandas_selecionadas,
    nome_pre: str,
    metodo_sel: str,
    k_bandas: int,
    output_dir,
    incluir_features_derivadas: bool = True,
):
    """
    Constrói a matriz Gold e salva como CSV.

    O sufixo do arquivo indica a configuração completa:
      gold_MSC_mi_k15_completo.csv    → bandas + features derivadas
      gold_MSC_mi_k15_espectros.csv   → somente intensidades de banda
      gold_RAW_baseline_completo.csv  → todas bandas + features derivadas
    """
    df_gold = build_gold_matrix(
        df_silver,
        bandas_selecionadas=bandas_selecionadas,
        incluir_features_derivadas=incluir_features_derivadas,
    )

    sufixo_sel = "baseline" if bandas_selecionadas is None else f"{metodo_sel}_k{k_bandas}"
    sufixo_feat = "completo" if incluir_features_derivadas else "espectros"
    nome_arquivo = f"gold_{nome_pre}_{sufixo_sel}_{sufixo_feat}.csv"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    caminho = output_dir / nome_arquivo

    df_gold.to_csv(caminho, index=False)
    n_feat = df_gold.shape[1] - 2
    print(f"  Gold salvo: {nome_arquivo} ({len(df_gold)} pixels × {n_feat} features)")
    return caminho


if __name__ == "__main__":
    from feature_selectors import selecionar_bandas_consenso

    ROOT_DIR   = Path(__file__).resolve().parents[1]
    DIR_SILVER = ROOT_DIR / "data" / "silver level" / "dados_calibrados"
    DIR_GOLD   = ROOT_DIR / "data" / "gold level"   / "features_extraidas"
    DIR_GOLD.mkdir(parents=True, exist_ok=True)

    K_BANDAS = 15

    print("=" * 60)
    print("ENGENHARIA DE ATRIBUTOS — PIXEL COMO AMOSTRA  k=15")
    print(f"Completo : {K_BANDAS} + 4 + {K_BANDAS-1} + 3 = {K_BANDAS*2+6} features")
    print(f"Espectros: {K_BANDAS} features")
    print("=" * 60)

    for arquivo in DIR_SILVER.glob("*.csv"):
        print(f"\n[{arquivo.name}]")
        df_silver = pd.read_csv(arquivo)
        nome_pre  = arquivo.stem.replace("master_", "").upper()

        for metodo_sel in ['mi', 'anova']:
            bandas = selecionar_bandas_consenso(df_silver, k=K_BANDAS, metodo=metodo_sel)
            salvar_gold(df_silver, bandas, nome_pre, metodo_sel, K_BANDAS, DIR_GOLD,
                        incluir_features_derivadas=True)
            salvar_gold(df_silver, bandas, nome_pre, metodo_sel, K_BANDAS, DIR_GOLD,
                        incluir_features_derivadas=False)

    print("\nExtração finalizada!")