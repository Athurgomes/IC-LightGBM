import numpy as np
from chemotools.derivative import SavitzkyGolay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def formatar_roi_para_2d(cubo_roi: np.ndarray, mask_2d: np.ndarray = None) -> np.ndarray:
    """Achata o cubo da ROI (H×W×bandas) em matriz [n_pixels × bandas] usando a máscara."""
    if mask_2d is None:
        mask_2d = cubo_roi[:, :, 0] > 0
    mask_bool = mask_2d.astype(bool)
    X = cubo_roi[mask_bool]
    if X.shape[0] == 0:
        return None
    return X


# =============================================================================
# PRÉ-PROCESSAMENTO ANTI-LEAKAGE — transformers sklearn (fit no TREINO de cada fold)
# =============================================================================
# Cada método de pré-processamento (MSC/SavGol/Z-Score) é um transformer cujo
# fit() estima qualquer parâmetro data-dependente apenas no treino, eliminando
# o vazamento. Uso: build_preprocessor(metodo).fit_transform(X_treino) e
# .transform(X_teste). Os pré-processamentos NÃO são precomputados no silver.


class MSCTransformer(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction com referência ajustada SÓ no treino.

    Mesmo método do pipeline original (referência = mediana dos espectros),
    porém a referência é estimada em fit() sobre o treino — sem ver o teste.

    Para cada espectro x: ajuste linear x ≈ a·ref + b (mínimos quadrados);
    espectro corrigido = (x − b) / a. Remove offset (b) e escala (a) de
    espalhamento, deixando a forma espectral comparável entre amostras.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.reference_ = np.median(X, axis=0)
        return self

    def transform(self, X):
        # MSC vetorizado: para cada espectro, ajuste OLS X_i ≈ a_i·ref + b_i e
        # correção (X_i - b_i)/a_i. Equivale a np.polyfit(ref, X_i, 1) por linha,
        # mas sem loop (rápido em centenas de milhares de pixels).
        X = np.asarray(X, dtype=float)
        ref = self.reference_
        ref_mean = ref.mean()
        ref_c = ref - ref_mean
        var_ref = float(ref_c @ ref_c) + 1e-12
        a = (X @ ref_c) / var_ref             # slope por espectro (n,)
        b = X.mean(axis=1) - a * ref_mean     # offset por espectro (n,)
        a_safe = np.where(np.abs(a) < self.eps, 1.0, a)
        out = (X - b[:, None]) / a_safe[:, None]
        deg = np.abs(a) < self.eps            # degenerado: só centra
        if deg.any():
            out[deg] = X[deg] - X[deg].mean(axis=1, keepdims=True)
        return out


def build_preprocessor(metodo: str):
    """Devolve um transformer sklearn para o método de pré-processamento.

    'raw'      → passthrough (RAW calibrado, linha de base)
    'msc'      → MSCTransformer (referência ajustada no treino)
    'savigol'  → Savitzky-Golay 1ª derivada (filtro fixo, sem parâmetros de treino)
    'z_score'  → StandardScaler (autoscaling por banda, ajustado no treino)

    Todos seguem o plano de trabalho; nenhum muda o método, apenas garantem
    que qualquer parâmetro data-dependente seja estimado só no treino.
    """
    m = (metodo or "raw").lower()
    if m == "raw":
        return FunctionTransformer()                      # identidade
    if m == "msc":
        return MSCTransformer()
    if m in ("savigol", "savgol", "sg"):
        return SavitzkyGolay(window_size=11, polynomial_order=3, derivate_order=1)
    if m in ("z_score", "zscore", "z-score"):
        return StandardScaler()
    raise ValueError(
        f"Método de pré-processamento '{metodo}' inválido. "
        "Use 'raw', 'msc', 'savigol' ou 'z_score'."
    )