"""
feature_selectors.py — Tópico 3: Seleção de Bandas

Implementa os três métodos de ranking de bandas descritos no plano de trabalho:
  1. Mutual Information (MI)
  2. Análise de Variância (ANOVA)
  3. Feature Importance do LightGBM (ganho por divisão)

E a seleção por consenso: bandas que aparecem no topo dos três rankings
são as mais confiáveis para distinguir cepas resistentes de sensíveis.

⚠️  REGRA DE USO — sem data leakage:
    Todas as funções devem ser chamadas APENAS com dados de TREINO.
    O modeling_roi.py garante isso ao chamar estas funções dentro
    de cada fold, antes de avaliar no conjunto de teste.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectFromModel
from lightgbm import LGBMClassifier


# =============================================================================
# CONSTANTES
# =============================================================================
COLUNAS_META = ['sample_id', 'target', 'block_id', 'roi_id', 'pixel_id']


def _get_band_cols(df):
    """Retorna as colunas de bandas espectrais (exclui metadados)."""
    return [c for c in df.columns if c not in COLUNAS_META]


# =============================================================================
# FASE 1 — RANKINGS INDIVIDUAIS
# Cada função retorna um DataFrame com (banda, score, rank)
# ordenado do mais relevante para o menos.
# =============================================================================

def rankear_bandas_mi(df_silver_treino):
    """
    Calcula o ranking de bandas por Mutual Information.

    O MI mede a dependência estatística entre cada banda e a classe alvo.
    Valores altos indicam que a banda carrega informação sobre resistência.

    Parâmetros
    ----------
    df_silver_treino : DataFrame silver somente com dados de TREINO

    Retorna
    -------
    DataFrame com colunas ['banda', 'score_mi', 'rank_mi'], ordenado por score_mi desc.
    """
    band_cols = _get_band_cols(df_silver_treino)
    X = df_silver_treino[band_cols].values
    y = df_silver_treino['target'].values

    scores = mutual_info_classif(X, y, random_state=42)

    df = pd.DataFrame({'banda': band_cols, 'score_mi': scores})
    df = df.sort_values('score_mi', ascending=False).reset_index(drop=True)
    df['rank_mi'] = df.index + 1
    return df


def rankear_bandas_anova(df_silver_treino):
    """
    Calcula o ranking de bandas por ANOVA (F-statistic).

    A ANOVA testa se a média de cada banda difere significativamente
    entre as classes resistente e sensível.

    Parâmetros
    ----------
    df_silver_treino : DataFrame silver somente com dados de TREINO

    Retorna
    -------
    DataFrame com colunas ['banda', 'score_anova', 'pvalor_anova', 'rank_anova'],
    ordenado por score_anova desc.
    """
    band_cols = _get_band_cols(df_silver_treino)
    X = df_silver_treino[band_cols].values
    y = df_silver_treino['target'].values

    f_scores, p_values = f_classif(X, y)

    df = pd.DataFrame({
        'banda':        band_cols,
        'score_anova':  f_scores,
        'pvalor_anova': p_values,
    })
    df = df.sort_values('score_anova', ascending=False).reset_index(drop=True)
    df['rank_anova'] = df.index + 1
    return df


def rankear_bandas_lightgbm(df_silver_treino):
    """
    Calcula o ranking de bandas por Feature Importance do LightGBM (ganho por divisão).

    Treina um LightGBM preliminar — o objetivo não é acurácia final, mas
    extrair quais bandas o modelo usa mais para separar as classes.

    Parâmetros
    ----------
    df_silver_treino : DataFrame silver somente com dados de TREINO

    Retorna
    -------
    DataFrame com colunas ['banda', 'score_lgbm', 'rank_lgbm'],
    ordenado por score_lgbm desc.
    """
    band_cols = _get_band_cols(df_silver_treino)
    X = df_silver_treino[band_cols].values
    y = df_silver_treino['target'].values

    lgbm = LGBMClassifier(
        n_estimators=100,
        num_leaves=7,
        max_depth=3,           # consistente com _build_model — evita overfitting nas médias de treino
        min_child_samples=2,
        lambda_l2=1.0,         # regularização L2 — consistente com _build_model
        class_weight='balanced',
        importance_type='gain',   # ganho por divisão, conforme o plano
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X, y)

    df = pd.DataFrame({
        'banda':      band_cols,
        'score_lgbm': lgbm.feature_importances_,
    })
    df = df.sort_values('score_lgbm', ascending=False).reset_index(drop=True)
    df['rank_lgbm'] = df.index + 1
    return df



# =============================================================================
# FASE 2 — TABELA DE COMPARAÇÃO DOS RANKINGS (MI + ANOVA)
# =============================================================================

def comparar_rankings(df_silver_treino):
    """
    Combina MI e ANOVA em uma tabela de consenso.

    O LightGBM FI foi removido do consenso: com apenas 3 amostras positivas,
    o modelo aprende ruído — seu ranking de bandas é instável e degrada a
    seleção (verificado empiricamente nos dados deste projeto).

    rankear_bandas_lightgbm() ainda está disponível para análise exploratória
    e para a célula de visualização do notebook.

    Retorna
    -------
    DataFrame com ['banda', 'score_mi', 'rank_mi', 'score_anova',
    'pvalor_anova', 'rank_anova', 'rank_medio'], ordenado por rank_medio asc.
    """
    df_mi    = rankear_bandas_mi(df_silver_treino)[['banda', 'score_mi', 'rank_mi']]
    df_anova = rankear_bandas_anova(df_silver_treino)[['banda', 'score_anova', 'pvalor_anova', 'rank_anova']]

    df = df_mi.merge(df_anova, on='banda')
    df['rank_medio'] = (df['rank_mi'] + df['rank_anova']) / 2
    df = df.sort_values('rank_medio').reset_index(drop=True)
    return df


# =============================================================================
# FASE 3 — SELEÇÃO DE BANDAS
# =============================================================================

def selecionar_bandas_consenso(df_silver_treino, k=10, metodo='rank_medio'):
    """
    Seleciona as k bandas mais relevantes para distinguir resistência.

    Parâmetros
    ----------
    df_silver_treino : DataFrame silver somente com dados de TREINO
    k                : número de bandas a selecionar
    metodo           : critério:
                       'rank_medio'  → média dos ranks MI+ANOVA (padrão)
                       'top_2_de_2'  → bandas no top-k de AMBOS MI e ANOVA
                       'mi'          → só MI (melhor resultado empírico neste dataset)
                       'anova'       → só ANOVA
                       'lgbm'        → só LightGBM FI (instável com N=19)

    Retorna
    -------
    lista com os nomes das k bandas selecionadas
    """
    if metodo == 'rank_medio':
        return comparar_rankings(df_silver_treino).head(k)['banda'].tolist()

    elif metodo == 'top_2_de_2':
        df = comparar_rankings(df_silver_treino)
        df['top_mi']    = df['rank_mi']    <= k
        df['top_anova'] = df['rank_anova'] <= k
        candidatas = df[df['top_mi'] & df['top_anova']].sort_values('rank_medio')
        if len(candidatas) < k:
            candidatas = df.head(k)
        return candidatas.head(k)['banda'].tolist()

    elif metodo == 'mi':
        return rankear_bandas_mi(df_silver_treino).head(k)['banda'].tolist()

    elif metodo == 'anova':
        return rankear_bandas_anova(df_silver_treino).head(k)['banda'].tolist()

    elif metodo == 'lgbm':
        return rankear_bandas_lightgbm(df_silver_treino).head(k)['banda'].tolist()

    else:
        raise ValueError(
            f"Método '{metodo}' inválido. "
            "Use 'rank_medio', 'top_2_de_2', 'mi', 'anova' ou 'lgbm'."
        )