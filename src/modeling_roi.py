"""
modeling_roi.py — Avaliação no nível de AMOSTRA/ROI (substitui o pixel-wise)

Por que este módulo existe
══════════════════════════════════════════════════════════════════════════════
O plano de trabalho define a matriz de modelagem como [n_amostras × n_bandas]
(um espectro por amostra). O pixel-wise foi um desvio introduzido na execução
que gera PSEUDO-REPLICAÇÃO: ~400k pixels altamente correlacionados tratados
como observações independentes inflam métricas e significância. A avaliação do
orientador pede explicitamente "redefinir unidade de análise: imagem/ROI > pixel".

Este módulo retorna à unidade correta:
  • Cada amostra → 1 espectro médio da ROI  →  matriz [19 × 226]
  • Validação por StratifiedKFold k=3 (cada amostra é predita uma vez)
  • Métricas calculadas sobre as 19 predições (não sobre pixels)
  • Pré-processamento e seleção de bandas ajustados DENTRO do fold (sem leakage)

Tudo segue o plano (MSC/SavGol/z-score, MI/ANOVA, LightGBM); as únicas
alterações a registrar são k-fold k=5 → k=3 (amostra) e a adição do teste de
permutação, ambas justificadas pelo N pequeno (19 amostras, 3 resistentes).

NOTA — por que NÃO Leave-One-Out: com modelo que prediz só a prevalência, o LOO
vaza o rótulo da amostra deixada de fora pela mudança de prevalência do treino
(AUC=1.0 até com rótulos embaralhados). O StratifiedKFold evita isso.

Funções principais
──────────────────
  load_roi_matrix()  → master_raw (pixels) + metadata → DataFrame [19 × 226]
  run_cv_amostra()   → StratifiedKFold k=3; predições + métricas no nível de amostra
  teste_permutacao() → distribuição nula de AUC/F1 e p-valor honesto
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from lightgbm import LGBMClassifier

# Imports robustos: funciona como `from src.modeling_roi import ...` (notebooks)
# e como `python src/modeling_roi.py` (execução direta).
try:
    from src.feature_engineering import build_gold_matrix
    from src.feature_selectors import selecionar_bandas_consenso
    from src.preprocessing import build_preprocessor
except ImportError:
    from feature_engineering import build_gold_matrix
    from feature_selectors import selecionar_bandas_consenso
    from preprocessing import build_preprocessor

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
MASTER_RAW = ROOT_DIR / "data" / "silver level" / "dados_calibrados" / "master_raw.csv"
METADATA = ROOT_DIR / "data" / "metadata.csv"

META_COLS = ("sample_id", "target")

# Hiperparâmetros padrão do LightGBM (avaliação sem otimização).
# Não fazem parte da grade; a otimização vem em otimizar_hiperparametros_roi.
DEFAULT_MODEL_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=7,
    max_depth=3,
    min_child_samples=2,
    lambda_l2=1.0,
    class_weight="balanced",
    importance_type="gain",
    random_state=42,
    verbose=-1,
)


# =============================================================================
# CARREGAMENTO — pixels → espectro médio por amostra (ROI)
# =============================================================================

def load_roi_matrix(master_raw=MASTER_RAW, metadata=METADATA) -> pd.DataFrame:
    """Agrega os pixels de cada amostra no espectro médio da ROI.

    O rótulo (`target`) vem do metadata.csv (fonte de verdade), não do nome
    do arquivo. Retorna DataFrame [n_amostras × n_bandas] com colunas
    exatamente [sample_id, b0..b{n-1}, target] — sem colunas extras, para não
    confundir o detector de bandas a jusante.
    """
    df_px = pd.read_csv(master_raw)
    band_cols = [c for c in df_px.columns if c not in ("sample_id", "target")]

    df_mean = df_px.groupby("sample_id", as_index=False)[band_cols].mean()

    meta = pd.read_csv(metadata)[["sample_id", "target"]]
    df_roi = df_mean.merge(meta, on="sample_id", how="left")

    if df_roi["target"].isna().any():
        faltando = df_roi.loc[df_roi["target"].isna(), "sample_id"].tolist()
        raise ValueError(f"Amostras sem rótulo no metadata.csv: {faltando}")

    df_roi["target"] = df_roi["target"].astype(int)
    return df_roi[["sample_id"] + band_cols + ["target"]]


# =============================================================================
# NÚCLEO — um fold (treino: n-1 amostras, teste: amostra(s) deixada(s) de fora)
# =============================================================================

def _band_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def _preprocessar_fold(df_tr, df_te, metodo_pre, band_cols):
    """Ajusta o pré-processamento NO TREINO e aplica em treino e teste."""
    pre = build_preprocessor(metodo_pre)
    Xtr = pre.fit_transform(df_tr[band_cols].values)
    Xte = pre.transform(df_te[band_cols].values)

    df_tr_pp = df_tr.copy()
    df_te_pp = df_te.copy()
    df_tr_pp[band_cols] = Xtr
    df_te_pp[band_cols] = Xte
    return df_tr_pp, df_te_pp


def _executar_fold_roi(df_tr, df_te, metodo_pre, metodo_sel, k,
                       model_params, incluir_derivadas):
    """Silver(ROI) → pré-proc(treino) → seleção(treino) → Gold → LGBM → prob."""
    band_cols = _band_cols(df_tr)

    df_tr_pp, df_te_pp = _preprocessar_fold(df_tr, df_te, metodo_pre, band_cols)

    # Seleção de bandas SÓ no treino (já no nível de amostra — sem N inflado)
    if metodo_sel is not None:
        bandas = selecionar_bandas_consenso(df_tr_pp, k=k, metodo=metodo_sel)
    else:
        bandas = None

    gold_tr = build_gold_matrix(df_tr_pp, bandas_selecionadas=bandas,
                                incluir_features_derivadas=incluir_derivadas)
    gold_te = build_gold_matrix(df_te_pp, bandas_selecionadas=bandas,
                                incluir_features_derivadas=incluir_derivadas)

    drop = [c for c in META_COLS if c in gold_tr.columns]
    X_tr, y_tr = gold_tr.drop(columns=drop), gold_tr["target"]
    X_te = gold_te.drop(columns=drop)

    model = LGBMClassifier(**model_params)
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]

    return pd.DataFrame({
        "sample_id": df_te["sample_id"].values,
        "y_true":    df_te["target"].values,
        "y_prob":    prob,
    })


# =============================================================================
# MÉTRICAS NO NÍVEL DE AMOSTRA
# =============================================================================

def _metricas_amostra(pred_df, limiar=0.5):
    y = pred_df["y_true"].values
    p = pred_df["y_prob"].values
    yp = (p >= limiar).astype(int)

    tem_2_classes = len(np.unique(y)) > 1
    cm = confusion_matrix(y, yp, labels=[0, 1])
    return {
        "n_amostras": int(len(y)),
        "n_resistentes": int(y.sum()),
        "Acuracia": accuracy_score(y, yp),
        "Precisao": precision_score(y, yp, zero_division=0),
        "Recall":   recall_score(y, yp, zero_division=0),
        "F1": f1_score(y, yp, zero_division=0),
        "ROC_AUC": roc_auc_score(y, p) if tem_2_classes else float("nan"),
        "matriz_confusao": cm,
    }


# =============================================================================
# PROTOCOLO — StratifiedKFold k=3 no nível de amostra
# =============================================================================
# Por que NÃO Leave-One-Out (LOO): com modelo pouco informativo (que prediz a
# prevalência), deixar de fora uma amostra POSITIVA muda a prevalência do treino
# (2 vs 3 positivas), e com class_weight='balanced' a constante predita muda
# conforme o rótulo da amostra deixada de fora → as predições "vazam" o próprio
# rótulo (AUC=1.0 até com rótulos embaralhados). O StratifiedKFold mantém a
# prevalência ~constante em todos os folds de treino, eliminando esse artefato.

def run_cv_amostra(df_roi, metodo_pre="msc", metodo_sel="mi", k=15,
                   model_params=None, incluir_derivadas=True, limiar=0.5,
                   n_splits=3, verbose=False):
    """StratifiedKFold (k=3) no nível de amostra. Predições por amostra + métricas.

    Cada amostra é predita exatamente uma vez (no fold em que cai no teste); as
    19 predições são acumuladas e as métricas calculadas uma vez sobre elas.
    A estratificação evita o vazamento de prevalência do LOO (ver nota acima).
    """
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y = df_roi["target"].values

    blocos = []
    for tr_idx, te_idx in skf.split(df_roi, y):
        df_tr = df_roi.iloc[tr_idx].reset_index(drop=True)
        df_te = df_roi.iloc[te_idx].reset_index(drop=True)
        if df_tr["target"].nunique() < 2:
            continue
        blocos.append(_executar_fold_roi(
            df_tr, df_te, metodo_pre, metodo_sel, k, model_params, incluir_derivadas
        ))

    pred_df = pd.concat(blocos, ignore_index=True)
    pred_df["y_pred"] = (pred_df["y_prob"] >= limiar).astype(int)
    metricas = _metricas_amostra(pred_df, limiar=limiar)

    if verbose:
        print(f"  Pré={metodo_pre} | Sel={metodo_sel} k={k} | derivadas={incluir_derivadas}")
        print(f"  F1={metricas['F1']:.3f} | ROC-AUC={metricas['ROC_AUC']:.3f} | "
              f"Recall={metricas['Recall']:.3f} | (N={metricas['n_amostras']}, "
              f"{metricas['n_resistentes']} resist.)")

    return {"metricas": metricas, "predicoes": pred_df}


# Alias de compatibilidade (notebooks antigos importam `run_logo`).
# O protocolo agora é StratifiedKFold k=3 (ver nota acima sobre o artefato do LOO).
run_logo = run_cv_amostra


# =============================================================================
# TESTE DE PERMUTAÇÃO — quanto do resultado é distinguível do acaso?
# =============================================================================

def teste_permutacao(df_roi, metodo_pre="msc", metodo_sel="mi", k=15,
                     model_params=None, incluir_derivadas=True,
                     metrica="ROC_AUC", n_perm=1000, random_state=42,
                     verbose=True):
    """Distribuição nula da métrica embaralhando os rótulos das 19 amostras.

    Para cada permutação, os rótulos são embaralhados NO NÍVEL DE AMOSTRA e a
    CV completa é refeita. O p-valor é a fração de permutações cuja métrica
    iguala ou supera a observada (com correção +1). Com 3 resistentes, espera-se
    que o resultado real caia próximo da nuvem nula — o p-valor quantifica isso.
    """
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS

    obs = run_cv_amostra(df_roi, metodo_pre, metodo_sel, k,
                         model_params, incluir_derivadas)["metricas"][metrica]

    rng = np.random.default_rng(random_state)
    y = df_roi["target"].values.copy()
    nulos = np.empty(n_perm)

    for i in range(n_perm):
        df_perm = df_roi.copy()
        df_perm["target"] = rng.permutation(y)
        nulos[i] = run_cv_amostra(df_perm, metodo_pre, metodo_sel, k,
                                  model_params, incluir_derivadas)["metricas"][metrica]
        if verbose and (i + 1) % max(1, n_perm // 10) == 0:
            print(f"    permutação {i + 1}/{n_perm}...")

    p_valor = (1 + int(np.sum(nulos >= obs))) / (n_perm + 1)

    if verbose:
        print(f"\n  {metrica} observado = {obs:.3f}")
        print(f"  Nula: média={np.nanmean(nulos):.3f} | "
              f"95º percentil={np.nanpercentile(nulos, 95):.3f}")
        print(f"  p-valor = {p_valor:.4f}  "
              f"({'NÃO' if p_valor > 0.05 else ''} distinguível do acaso a 5%)")

    return {"observado": obs, "p_valor": p_valor, "nulos": nulos,
            "n_perm": n_perm, "metrica": metrica}


# =============================================================================
# OTIMIZAÇÃO DE HIPERPARÂMETROS — Nested CV no nível de AMOSTRA
# =============================================================================
# Nested CV no nível de amostra: outer StratifiedKFold k=3 (cada fold testa
# ~6 amostras → métricas por fold continuam definidas) e inner GridSearchCV.
# O inner cai para k=2 porque, com apenas 3 resistentes, o treino externo tem
# 2 positivas — alteração documentada (análoga ao k=5→k=3 do parcial). Com mais
# dados, aumente n_splits_inner/outer (o inner adapta-se à classe minoritária).

# Grades de hiperparâmetros (cobrem os params exigidos pelo plano: num_leaves,
# max_depth, learning_rate, feature_fraction, lambda_l1/l2, min_data_in_leaf
# = min_child_samples, boosting_type).
PARAM_GRID_REDUZIDA = {
    "boosting_type":     ["gbdt", "dart"],
    "num_leaves":        [7, 31],
    "max_depth":         [3, -1],
    "learning_rate":     [0.01, 0.1],
    "feature_fraction":  [0.7, 1.0],
    "lambda_l1":         [0.0, 1.0],
    "lambda_l2":         [0.1, 1.0],
    "min_child_samples": [2, 10],
}

PARAM_GRID_COMPLETA = {
    "boosting_type":     ["gbdt", "dart"],
    "num_leaves":        [7, 15, 31],
    "max_depth":         [3, 5, -1],
    "learning_rate":     [0.01, 0.05, 0.1],
    "feature_fraction":  [0.6, 0.8, 1.0],
    "lambda_l1":         [0.0, 0.1, 1.0],
    "lambda_l2":         [0.1, 1.0, 5.0],
    "min_child_samples": [2, 5, 10],
}

# Grade conservadora para baseline+derivadas (muitas features, alto overfitting).
PARAM_GRID_BASELINE_DERIVADAS = {
    "boosting_type":     ["gbdt", "dart"],
    "num_leaves":        [7, 15],
    "max_depth":         [3, 5],
    "learning_rate":     [0.01, 0.05],
    "feature_fraction":  [0.5, 0.7],
    "lambda_l1":         [0.0, 1.0],
    "lambda_l2":         [1.0, 5.0],
    "min_child_samples": [5, 10],
}


def _build_base_lgbm():
    """Parâmetros fixos do LightGBM; o resto é varrido pelo GridSearch."""
    return LGBMClassifier(
        n_estimators=100, class_weight="balanced", importance_type="gain",
        random_state=42, verbose=-1,
    )


def otimizar_hiperparametros_roi(
    df_roi, metodo_pre="msc", metodo_selecao="mi", k_bandas=15,
    param_grid=None, incluir_features_derivadas=False,
    n_splits_outer=3, n_splits_inner=2, verbose=True,
):
    """Nested CV no nível de amostra. Retorna dict compatível com o NB07.

    Chaves do retorno: melhores_params, melhor_score_cv, metricas_teste,
    metricas_agregadas, metricas_por_fold, params_por_fold, scores_cv_por_fold,
    df_cv_resultados, grid_search, feature_names, bandas.
    """
    if param_grid is None:
        raise ValueError("Forneça param_grid (ex.: PARAM_GRID_REDUZIDA).")

    band_cols = _band_cols(df_roi)
    y_all = df_roi["target"].values
    cv_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

    fold_metricas, fold_params, fold_scores_cv = [], [], []
    best_fold_score, best_grid, best_df_cv = -1, None, None
    feature_names, bandas_ref = None, None

    for fold_idx, (tr_idx, te_idx) in enumerate(cv_outer.split(df_roi, y_all)):
        df_tr = df_roi.iloc[tr_idx].reset_index(drop=True)
        df_te = df_roi.iloc[te_idx].reset_index(drop=True)
        if df_tr["target"].nunique() < 2:
            continue

        # Pré-proc (fit no treino) + seleção de bandas (treino) — sem leakage no outer
        df_tr_pp, df_te_pp = _preprocessar_fold(df_tr, df_te, metodo_pre, band_cols)
        bandas = (selecionar_bandas_consenso(df_tr_pp, k=k_bandas, metodo=metodo_selecao)
                  if metodo_selecao is not None else None)
        bandas_ref = bandas

        gold_tr = build_gold_matrix(df_tr_pp, bandas_selecionadas=bandas,
                                    incluir_features_derivadas=incluir_features_derivadas)
        gold_te = build_gold_matrix(df_te_pp, bandas_selecionadas=bandas,
                                    incluir_features_derivadas=incluir_features_derivadas)
        drop = [c for c in META_COLS if c in gold_tr.columns]
        X_tr, y_tr = gold_tr.drop(columns=drop), gold_tr["target"].values
        X_te, y_te = gold_te.drop(columns=drop), gold_te["target"].values
        if feature_names is None:
            feature_names = list(X_tr.columns)

        # Inner k limitado pela classe minoritária do treino externo
        n_min = int(min(np.bincount(y_tr)))
        inner_k = max(2, min(n_splits_inner, n_min))
        cv_inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=42)

        grid = GridSearchCV(_build_base_lgbm(), param_grid=param_grid,
                            cv=cv_inner, scoring="f1", n_jobs=-1, refit=True)
        grid.fit(X_tr, y_tr)

        y_pred = grid.best_estimator_.predict(X_te)
        y_prob = grid.best_estimator_.predict_proba(X_te)[:, 1]
        metricas = {
            "Acurácia": accuracy_score(y_te, y_pred),
            "Precisão": precision_score(y_te, y_pred, zero_division=0),
            "Recall":   recall_score(y_te, y_pred, zero_division=0),
            "F1-score": f1_score(y_te, y_pred, zero_division=0),
            "ROC-AUC":  roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5,
        }
        fold_metricas.append(metricas)
        fold_params.append(grid.best_params_)
        fold_scores_cv.append(grid.best_score_)

        if verbose:
            print(f"  Fold externo {fold_idx+1}/{n_splits_outer} "
                  f"(treino {len(df_tr)} am., teste {len(df_te)} am., inner k={inner_k}): "
                  f"F1cv={grid.best_score_:.3f} | F1ext={metricas['F1-score']:.3f} "
                  f"| AUCext={metricas['ROC-AUC']:.3f}")

        if grid.best_score_ > best_fold_score:
            best_fold_score, best_grid = grid.best_score_, grid
            dfcv = pd.DataFrame(grid.cv_results_)
            cols = ([c for c in dfcv.columns if c.startswith("param_")]
                    + ["mean_test_score", "std_test_score", "rank_test_score"])
            best_df_cv = dfcv[cols].sort_values("rank_test_score").reset_index(drop=True)

    if not fold_metricas:
        raise RuntimeError("Nenhum fold válido — dataset pequeno demais.")

    df_met = pd.DataFrame(fold_metricas)
    agreg = {c: df_met[c].mean() for c in df_met.columns}
    agreg.update({f"{c} (std)": df_met[c].std() for c in df_met.columns})
    agreg["Folds válidos"] = len(fold_metricas)

    melhor_idx = int(np.argmax(fold_scores_cv))
    params_venc = fold_params[melhor_idx]

    if verbose:
        print(f"\n  Melhores params (fold {melhor_idx+1}, F1cv={fold_scores_cv[melhor_idx]:.3f}): "
              f"{params_venc}")
        print(f"  Métricas externas médias: F1={agreg['F1-score']:.3f} "
              f"± {agreg['F1-score (std)']:.3f} | AUC={agreg['ROC-AUC']:.3f}")

    return {
        "melhores_params":    params_venc,
        "melhor_score_cv":    fold_scores_cv[melhor_idx],
        "metricas_teste":     {k: v for k, v in agreg.items()
                               if "(std)" not in k and k != "Folds válidos"},
        "metricas_agregadas": agreg,
        "metricas_por_fold":  fold_metricas,
        "params_por_fold":    fold_params,
        "scores_cv_por_fold": fold_scores_cv,
        "df_cv_resultados":   best_df_cv,
        "grid_search":        best_grid,
        "feature_names":      feature_names,
        "bandas":             bandas_ref,
    }


# =============================================================================
# DEMO / VERIFICAÇÃO
# =============================================================================

def main():
    print("=" * 64)
    print("AVALIAÇÃO NO NÍVEL DE AMOSTRA (ROI) — StratifiedKFold k=3")
    print("=" * 64)

    df_roi = load_roi_matrix()
    print(f"Matriz ROI: {df_roi.shape[0]} amostras × "
          f"{df_roi.shape[1] - 2} bandas | "
          f"{int(df_roi['target'].sum())} resistentes\n")

    # Compara os 4 pré-processamentos do plano, seleção MI k=15, com derivadas.
    print("K-Fold k=3 (amostra) por pré-processamento (MI, k=15, derivadas=True):")
    for pre in ["raw", "msc", "savigol", "z_score"]:
        run_cv_amostra(df_roi, metodo_pre=pre, metodo_sel="mi", k=15,
                       incluir_derivadas=True, verbose=True)
        print()

    # Teste de permutação (demo com n_perm reduzido; final deve usar 1000).
    print("-" * 64)
    print("TESTE DE PERMUTAÇÃO (MSC, MI k=15) — demo n_perm=100")
    print("-" * 64)
    teste_permutacao(df_roi, metodo_pre="msc", metodo_sel="mi", k=15,
                     metrica="ROC_AUC", n_perm=100)


if __name__ == "__main__":
    try:                                  # acentos legíveis no console Windows
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.path.insert(0, str(ROOT_DIR / "src"))
    main()
