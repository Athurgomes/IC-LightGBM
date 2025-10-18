import matplotlib.pyplot as plt
import numpy as np
import spectral

def plotar_imagem_rgb(img, bandas):
    try:
        plt.figure(figsize=(10, 8))
        spectral.imshow(img, bands=bandas)
        plt.title(f'Imagem Hiperespectral (RGB Falso)\nBandas R={bandas[0]}, G={bandas[1]}, B={bandas[2]}', fontsize=14)
        plt.xlabel('Pixels (Coordenada X)', fontsize=12)
        plt.ylabel('Pixels (Coordenada Y)', fontsize=12)
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar: {e}")

def plotar_media_espectral(img):
    try:
        hsi_cube=img.load()
        mean_spectrum=np.mean(hsi_cube, axis=(0, 1))
        std_spectrum=np.std(hsi_cube, axis=(0, 1))
        wavelengths=img.bands.centers
        plt.figure(figsize=(12, 7))
        plt.plot(wavelengths, mean_spectrum, label='Média Espectral', color='blue', linewidth=2)
        plt.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color='blue', alpha=0.2, label='±1σ (Média)')
        plt.title('Assinatura Espectral Média', fontsize=16)
        plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
        plt.ylabel('Intensidade', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar: {e}")

def plotar_histograma_banda(img, banda_idx=150):
    try:
        hsi_cube=img.load()
        banda=hsi_cube[:, :, banda_idx].ravel()
        plt.figure(figsize=(10, 6))
        plt.hist(banda, bins=100, color='gray')
        plt.title(f"Histograma - Banda {banda_idx} (Dados Brutos)", fontsize=16)
        plt.xlabel("Intensidade", fontsize=12)
        plt.ylabel("Contagem de Pixels", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar {banda_idx}: {e}")

    print(f"Formato do cubo (shape): {hsi_cube.shape}")
    print(f"Valor médio de todos os pixels: {hsi_cube.mean():.6f}")
    print(f"Soma de todos os pixels: {hsi_cube.sum():.2f}")
    print(f"Desvio padrão: {hsi_cube.std():.2f}")