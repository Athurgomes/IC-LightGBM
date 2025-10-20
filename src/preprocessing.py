import numpy as np
from chemotools.scatter import MultiplicativeScatterCorrection
from chemotools.derivative import SavitzkyGolay
from sklearn.preprocessing import StandardScaler 

def apply_msc(X: np.ndarray) -> np.ndarray:

    msc = MultiplicativeScatterCorrection()
    X_msc = msc.fit_transform(X)
    return X_msc

def apply_savitzky_golay(X: np.ndarray, window_size: int = 21, poly_order: int = 3, deriv_order: int = 1) -> np.ndarray:
    if window_size <= poly_order:
        poly_order = window_size - 1
    if window_size % 2 == 0:
        window_size += 1

    sg = SavitzkyGolay(
        window_size=window_size,
        polynomial_order=poly_order,
        derivate_order=deriv_order
    )
    X_sg = sg.fit_transform(X)
    return X_sg

def apply_zscore(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def create_preprocessing_pipeline(X: np.ndarray) -> np.ndarray:
    X_msc = apply_msc(X)
    X_sg = apply_savitzky_golay(
        X_msc, 
        window_size=11,
        poly_order=2,
        deriv_order=1
    )
    X_gold = apply_zscore(X_sg)
    print("Pipeline concluído (MSC -> SG1 -> Z-Score).")
    return X_gold