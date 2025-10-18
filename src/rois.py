import spectral
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rois(img, bandas=(150,100,50)):
    rgb=spectral.get_rgb(img, bandas)
    gray=cv2.cvtColor((rgb*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100,
        param1=100, param2=30,
        minRadius=50, maxRadius=200)
    mask=np.zeros(gray.shape, dtype=np.uint8)
    x, y, r = circles[0][0]
    x, y, r = int(x), int(y), int(r)
    imagem_com_circulo = (rgb * 255).astype(np.uint8).copy()
    cv2.circle(imagem_com_circulo, (x, y), r, (255, 0, 0), 3)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 1, thickness=-1)
    mask3d = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
    roi = img * mask3d
    roi_rgb = spectral.get_rgb(roi, bandas)
    axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(imagem_com_circulo)
    axes[0].set_title('ROI Detectado na Imagem Original')
    axes[0].axis('off')
    axes[1].imshow(roi_rgb)
    axes[1].set_title('Apenas o ROI Extraído')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
