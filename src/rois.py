import spectral
import cv2
import numpy as np
import matplotlib.pyplot as plt

def selecionar_rois(img, bandas=(135,85,35)):
    #Argumentos:
        #img (np.array): O cubo de dados hiperespectral.
        #bandas (tuple): As bandas para usar na visualização RGB (para detecção).

    #Retorna:
        #roi (np.array): O cubo hiperespectral contendo apenas os dados do ROI.
        #rgb (np.array): A imagem RGB original (para visualização).
        #circulo (tuple): As coordenadas (x, y, r) do círculo detectado.  
    rgb = spectral.get_rgb(img, bandas)
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=50, maxRadius=200)
    if circles is None:
        print("Nenhum círculo foi detectado.")
        return None, rgb, None
    x, y, r = circles[0][0]
    x, y, r = int(x), int(y), int(r)
    circulo_coords = (x, y, r)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 1, thickness=-1)
    mask3d = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
    roi = img * mask3d
    return roi, rgb, circulo_coords

def plotar_comparacao_roi(roi_extraido, rgb_original, circulo, bandas=(150,100,50)):
    (x, y, r) = circulo
    imagem_com_circulo = (rgb_original * 255).astype(np.uint8).copy()
    cv2.circle(imagem_com_circulo, (x, y), r, (255, 0, 0), 3) 
    roi_rgb = spectral.get_rgb(roi_extraido, bandas)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(imagem_com_circulo)
    axes[0].set_title('ROI Detectado na Imagem Original')
    axes[0].axis('off')
    axes[1].imshow(roi_rgb)
    axes[1].set_title('Apenas o ROI Extraído')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

def plotar_media_espectral(roi_array, wavelengths):
    try:
        mask_2d = roi_array[:, :, 0] > 0
        pixels_do_roi = roi_array[mask_2d]
        if pixels_do_roi.shape[0] == 0:
            print("Erro: O ROI está vazio ou não contém dados.")
            return
        mean_spectrum = np.mean(pixels_do_roi, axis=0)
        std_spectrum = np.std(pixels_do_roi, axis=0)
        plt.figure(figsize=(12, 7))
        plt.plot(wavelengths, mean_spectrum, label='Média Espectral (ROI)', color='blue', linewidth=2)
        plt.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color='blue', alpha=0.2, label='±1σ')
        plt.title('Assinatura Espectral Média do ROI (Com remoção de bandas)', fontsize=16)
        plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
        plt.ylabel('Intensidade', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Erro ao plotar: {e}")
