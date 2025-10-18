import spectral
import cv2
import numpy as np
import matplotlib.pyplot as plt

def selecionar_rois(img, bandas=(150,100,50)):
    """
    Detecta e extrai o principal ROI circular de uma imagem hiperespectral.

    Argumentos:
        img (np.array): O cubo de dados hiperespectral.
        bandas (tuple): As bandas para usar na visualização RGB (para detecção).

    Retorna:
        roi (np.array): O cubo hiperespectral contendo apenas os dados do ROI.
        rgb (np.array): A imagem RGB original (para visualização).
        circulo (tuple): As coordenadas (x, y, r) do círculo detectado.
    """
    
    # 1. Obter RGB e escala de cinza para detecção
    rgb = spectral.get_rgb(img, bandas)
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 2. Detectar círculos
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=50, maxRadius=200)
    
    # Adicionando uma verificação para evitar erros se nenhum círculo for encontrado
    if circles is None:
        print("Aviso: Nenhum círculo foi detectado.")
        return None, rgb, None

    # 3. Obter coordenadas do círculo
    x, y, r = circles[0][0]
    x, y, r = int(x), int(y), int(r)
    circulo_coords = (x, y, r)
    
    # 4. Criar a máscara 3D
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 1, thickness=-1)
    mask3d = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
    
    # 5. Extrair o ROI usando a máscara
    roi = img * mask3d
    
    # 6. Retornar os dados necessários para a plotagem
    return roi, rgb, circulo_coords

def plotar_comparacao_roi(roi_extraido, rgb_original, circulo, bandas=(150,100,50)):
    """
    Plota a comparação entre a imagem original com o ROI detectado
    e a imagem do ROI extraído.
    """
    
    # 1. Preparar a imagem original com o círculo desenhado
    (x, y, r) = circulo
    imagem_com_circulo = (rgb_original * 255).astype(np.uint8).copy()
    cv2.circle(imagem_com_circulo, (x, y), r, (255, 0, 0), 3) # (Vermelho, 3px de espessura)
    
    # 2. Obter o RGB do ROI que foi extraído
    roi_rgb = spectral.get_rgb(roi_extraido, bandas)
    
    # 3. Plotar as duas imagens lado a lado
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
    """
    Calcula e plota a média e mediana espectral de um ROI extraído 
    (array NumPy), ignorando os pixels zerados (mascarados).
    """
    try:
        # --- Esta é a lógica chave ---
        # 1. Encontrar todos os pixels que NÃO são zero (fora da máscara)
        #    Usamos a primeira banda (índice 0) como referência.
        mask_2d = roi_array[:, :, 0] > 0
        
        # 2. Selecionar apenas os pixels que estão DENTRO do ROI
        #    Isso transforma o array 3D em 2D: (N_pixels_no_roi, N_bandas)
        pixels_do_roi = roi_array[mask_2d]

        # 3. Verificar se o ROI não está vazio
        if pixels_do_roi.shape[0] == 0:
            print("Erro: O ROI está vazio ou não contém dados.")
            return

        # 4. Calcular estatísticas apenas nesses pixels (axis=0)
        mean_spectrum = np.mean(pixels_do_roi, axis=0)
        std_spectrum = np.std(pixels_do_roi, axis=0)
        
        # --- O resto é o seu código de plotagem ---
        plt.figure(figsize=(12, 7))
        
        # Plot da Média
        plt.plot(wavelengths, mean_spectrum, label='Média Espectral (ROI)', color='blue', linewidth=2)
        # Plot do Desvio Padrão
        plt.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, 
                         color='blue', alpha=0.2, label='±1σ (Média ROI)')
        
        plt.title('Assinatura Espectral Média do ROI', fontsize=16)
        plt.xlabel('Comprimento de Onda (nm)', fontsize=12)
        plt.ylabel('Intensidade', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Erro ao plotar: {e}")
