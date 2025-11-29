import numpy as np

def remover_bandas_laterais(cubo: np.ndarray, bandas_inicio: int, bandas_fim: int) -> np.ndarray:
    try:
        num_bandas_original = cubo.shape[2]
        if num_bandas_original <= (bandas_inicio + bandas_fim):
            print(f"  AVISO: Cubo tem {num_bandas_original} bandas. "
                  f"Insuficiente para remover {bandas_inicio + bandas_fim}.")
            return None
        cubo_cortado = cubo[:, :, bandas_inicio:-bandas_fim]
        return cubo_cortado
    except Exception as e:
        print(f"  ERRO ao remover bandas: {e}")
        return None