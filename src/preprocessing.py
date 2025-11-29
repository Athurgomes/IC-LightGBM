import numpy as np
from chemotools.scatter import MultiplicativeScatterCorrection
from chemotools.derivative import SavitzkyGolay
from sklearn.preprocessing import StandardScaler 

def formatar_roi_para_2d(cubo_roi: np.ndarray) -> np.ndarray:
    mask_2d = cubo_roi[:, :, 0] > 0
    X = cubo_roi[mask_2d]
    if X.shape[0] == 0:
        return None
    return X

def apply_msc(X: np.ndarray) -> np.ndarray:
    mediana = np.median(X, axis=0)
    msc_mediana = MultiplicativeScatterCorrection(reference=mediana)
    return msc_mediana.fit_transform(X)

def apply_savitzky_golay(X: np.ndarray, window_size: int = 11, poly_order: int = 3, deriv_order: int = 1) -> np.ndarray:
    if window_size <= poly_order:
        poly_order = window_size - 1
    if window_size % 2 == 0:
        window_size += 1
    sg = SavitzkyGolay(window_size=window_size, polynomial_order=poly_order, derivate_order=deriv_order)
    return sg.fit_transform(X)

def apply_zscore(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)