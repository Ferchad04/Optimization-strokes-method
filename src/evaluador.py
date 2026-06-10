import os
import torch
import math
from PIL import Image
import torchvision.transforms as transforms
from src.metricas_error import MetricasReconstruccion

def cargar_imagen(ruta, alto=400, ancho=500, dispositivo='cpu'):
    """Carga y normaliza una imagen para evaluación."""
    if not os.path.exists(ruta):
        return None
    imagen = Image.open(ruta).convert("RGB")
    transformacion = transforms.Compose([
        transforms.Resize((alto, ancho)),
        transforms.ToTensor()
    ])
    return transformacion(imagen).to(dispositivo)

def calcular_psnr(mse):
    """
    Calcula el Peak Signal-to-Noise Ratio (PSNR).
    Métrica estándar en compresión y reconstrucción de imágenes.
    """
    if mse == 0:
        return float('inf')
    # El valor máximo del píxel es 1.0 (imágenes normalizadas)
    max_pixel = 1.0
    return 20 * math.log10(max_pixel) - 10 * math.log10(mse)

def evaluar_resultados():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluador_base = MetricasReconstruccion(tipo_reduccion='mean').to(dispositivo)
    
    ruta_original = "data/pintura_proyecto.jpeg"
    original = cargar_imagen(ruta_original, dispositivo=dispositivo)
    
    if original is None:
        print(f"Error: No se encontró la imagen original en {ruta_original}")
        return

    # Diccionario con los nombres de los archivos generados por cada configuración
    archivos_prueba = {
        "Config 1 (Circulo)": "data/Circulos_01/Circulos.png",
        "Config 2 (Elipse)": "data/elipse_02/reconstruccion_900.png",
        "Config 3 (Gaussiana)": "data/gaussiana_01/gaussiana.png",
        "Config 4 (Rectangulo)": "data/rectangulo_01/reconstruccion_1000.png",
        "Config 5 (Capsula)": "data/capsula_01/capsula.png"
    }

    print("-" * 80)
    print(f"{'Configuración':<25} | {'MSE (↓)':<10} | {'MAE (↓)':<10} | {'PSNR (↑)':<10} | {'SSIM (↑)':<10}")
    print("-" * 80)

    with torch.no_grad():
        for nombre, ruta in archivos_prueba.items():
            reconstruccion = cargar_imagen(ruta, dispositivo=dispositivo)
            
            if reconstruccion is None:
                print(f"{nombre:<25} | Archivo no encontrado")
                continue
            
            # Cálculos
            mse_val = evaluador_base.mse(original, reconstruccion).item()
            mae_val = evaluador_base.mae(original, reconstruccion).item()
            ssim_loss = evaluador_base.perdida_ssim(original, reconstruccion).item()
            
            # Derivaciones
            ssim_val = 1.0 - ssim_loss # Convertir pérdida a índice de similitud
            psnr_val = calcular_psnr(mse_val)
            
            print(f"{nombre:<25} | {mse_val:.6f}   | {mae_val:.6f}   | {psnr_val:>6.2f} dB  | {ssim_val:.4f}")

    print("-" * 80)
    print("Indicadores: (↓) Menor es mejor, (↑) Mayor es mejor.")

if __name__ == "__main__":
    evaluar_resultados()