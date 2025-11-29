import spectral
import numpy as np

def carreagar_dados(path_raw: str, path_white: str, path_dark: str):
    try:
        raw_cube = spectral.open_image(path_raw).load()
        white_cube = spectral.open_image(path_white).load()
        dark_cube = spectral.open_image(path_dark).load()
        wavelengths = raw_cube.bands.centers
    except Exception as e:
        print(f"  Erro ao carregar os dados: {e}")
        return None, None
    try:
        dark_spec_mean = np.mean(dark_cube, axis=(0, 1))
        white_spec_mean = np.mean(white_cube, axis=(0, 1))
        denominador = white_spec_mean - dark_spec_mean
        numerador = raw_cube - dark_spec_mean
        reflectancia_cube = np.clip(numerador / denominador, 0, 1.5)
        return reflectancia_cube, wavelengths
    except Exception as e:
        print(f"  Erro durante a calibração: {e}")
        return None, None