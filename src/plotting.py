import matplotlib.pyplot as plt
import numpy as np
import spectral

def plotar_imagem_rgb(img, bandas=(150, 100, 50)):
    """
    Plota uma imagem hiperespectral em falsa-cor (RGB).

    Args:
        img (spectral.Image): O objeto de imagem hiperespectral carregado.
        bandas (tuple): Uma tupla com os 3 índices de banda para R, G e B.
    """
    try:
        print("Gerando a visualização da imagem original (RGB)...")
        plt.figure(figsize=(10, 8))
        
        spectral.imshow(img, bands=bandas)
        
        plt.title(f'Imagem Hiperespectral (Falsa Cor)\nBandas R={bandas[0]}, G={bandas[1]}, B={bandas[2]}', fontsize=14)
        plt.xlabel('Pixels (Coordenada X)', fontsize=12)
        plt.ylabel('Pixels (Coordenada Y)', fontsize=12)
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar a imagem RGB: {e}")

def plotar_media_espectral(img):
    """
    Calcula e plota a assinatura espectral média de uma imagem hiperespectral.

    Args:
        img (spectral.Image): O objeto de imagem hiperespectral carregado.
    """
    try:
        print("\nCalculando e gerando o gráfico da assinatura espectral média...")
        
        # Carrega o cubo para a memória para fazer o cálculo
        hsi_cube = img.load()
        
        # Calcula a média ao longo das dimensões espaciais (linhas e colunas)
        mean_spectrum = np.mean(hsi_cube, axis=(0, 1))
        
        # Obtém os comprimentos de onda do objeto de imagem
        wavelengths = img.bands.centers
        
        plt.figure(figsize=(12, 7))
        plt.plot(wavelengths, mean_spectrum, label='Média Espectral da Imagem', color='blue')
        
        plt.title('Assinatura Espectral Média', fontsize=16)
        plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
        plt.ylabel('Intensidade / Reflectância Média', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar a média espectral: {e}")