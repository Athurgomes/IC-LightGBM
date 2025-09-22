import matplotlib.pyplot as plt
import spectral
import numpy as np

def plotar_comparacao_rgb(lista_imagens, lista_nomes, bandas=(200, 150, 50)):
    if len(lista_imagens) != 2:
        print("Esta função foi projetada para comparar exatamente duas imagens.")
        return
    try:
        fig, axes=plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Comparação Visual das Amostras (RGB)', fontsize=18)
        for i, ax in enumerate(axes):
            rgb=spectral.get_rgb(lista_imagens[i], bands=bandas)
            ax.imshow(rgb)
            ax.set_title(lista_nomes[i], fontsize=14)
            ax.set_xlabel('Pixels (X)')
        axes[0].set_ylabel('Pixels (Y)')
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar a comparação RGB: {e}")

def plotar_comparacao_estatistica_espectral(lista_imagens, lista_nomes):
    if len(lista_imagens)!=2:
        print("Esta função foi projetada para comparar exatamente duas imagens.")
        return
    try:
        fig, axes=plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        fig.suptitle('Análise Espectral Comparativa Detalhada', fontsize=18)
        colors = ['blue', 'green']
        for i, ax in enumerate(axes):
            hsi_cube=lista_imagens[i].load()
            mean_spec=np.mean(hsi_cube, axis=(0, 1))
            median_spec=np.median(hsi_cube, axis=(0, 1))
            std_spec=np.std(hsi_cube, axis=(0, 1))
            wavelengths=lista_imagens[i].bands.centers
            ax.plot(wavelengths, mean_spec, label='Média Espectral', color=colors[i], linewidth=2)
            ax.plot(wavelengths, median_spec, label='Mediana Espectral', color='red', linestyle='--')
            ax.fill_between(wavelengths, mean_spec - std_spec, mean_spec + std_spec, color=colors[i], alpha=0.2, label='±1σ')
            ax.set_title(lista_nomes[i], fontsize=14)
            ax.set_xlabel("Comprimento de Onda (nm)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
        axes[0].set_ylabel("Intensidade")
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar a comparação estatística: {e}")

def plotar_comparacao_histogramas(lista_imagens, lista_nomes, banda_idx):
    if len(lista_imagens)!=2:
        print("Esta função foi projetada para comparar exatamente duas imagens.")
        return
    try:
        fig, axes=plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        fig.suptitle(f'Comparação dos Histogramas da Banda {banda_idx}', fontsize=18)
        for i, ax in enumerate(axes):
            hsi_cube=lista_imagens[i].load()
            banda=hsi_cube[:, :, banda_idx].ravel()
            ax.hist(banda, bins=100, color='gray')
            ax.set_title(lista_nomes[i], fontsize=14)
            ax.set_xlabel("Intensidade")
            ax.grid(True, linestyle='--', alpha=0.6)
        axes[0].set_ylabel("Contagem de Pixels")
            #ax.set_ylabel("Contagem de Pixels")
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar a comparação de histogramas: {e}")