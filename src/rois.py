import spectral
import cv2
import numpy as np

def selecionar_rois(img: np.ndarray, bandas: tuple, param2_ajustado: int):
    rgb_original = None
    circulo_coords = None
    try:
        rgb = spectral.get_rgb(img, bandas)
        rgb_original = (rgb * 255).astype(np.uint8).copy()
    except IndexError:
        print(f"  ERRO DE ÍNDICE: Bandas {bandas} fora dos limites.")
        return None, None, None
    gray = cv2.cvtColor(rgb_original, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,          
        minDist=100,     
        param1=50,      
        param2=param2_ajustado, 
        minRadius=50,    
        maxRadius=200    
    )
    if circles is None:
        return None, rgb_original, None   
    x, y, r = circles[0][0] 
    x, y, r = int(x), int(y), int(r)
    r = int(r * 0.95)
    circulo_coords = (x, y, r)
    cv2.circle(rgb_original, (x, y), r, (255, 0, 0), 3)
    mask_2d = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask_2d, (x, y), r, 1, thickness=-1)
    mask_3d = np.repeat(mask_2d[:, :, np.newaxis], img.shape[2], axis=2)
    roi = img * mask_3d
    return roi, rgb_original, circulo_coords