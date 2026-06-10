import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from src.pinceladas_lote import LotePinceladasDiferenciables
from src.perdida_compuesta import PerdidaCompuesta
from src.motor_optimizacion import OptimizadorStrokes

def cargar_imagen_objetivo(ruta_imagen, alto, ancho, dispositivo):
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")
        
    imagen = Image.open(ruta_imagen).convert("RGB")
    transformacion = transforms.Compose([
        transforms.Resize((alto, ancho)),
        transforms.ToTensor()
    ])
    return transformacion(imagen).to(dispositivo)

def generar_lote_candidato(imagen_objetivo, lienzo_actual, alto, ancho, K):
    """
    Genera K posiciones iniciales utilizando un muestreo multinomial 
    ponderado por la magnitud del error absoluto.
    """
    with torch.no_grad():
        mapa_error = torch.mean(torch.abs(imagen_objetivo - lienzo_actual), dim=0).flatten()
        
        # Normalizar para obtener una distribución de probabilidad
        probabilidades = mapa_error / mapa_error.sum()
        
        # Seleccionar K índices sin reemplazo
        indices_k = torch.multinomial(probabilidades, K, replacement=False)
        
        y_iniciales = torch.div(indices_k, ancho, rounding_mode='trunc').cpu().numpy()
        x_iniciales = (indices_k % ancho).cpu().numpy()
        
        colores_rgb_iniciales = []
        for i in range(K):
            color = imagen_objetivo[:, y_iniciales[i], x_iniciales[i]].cpu().numpy()
            colores_rgb_iniciales.append(color)

    angulos_iniciales = np.zeros(K)
    alfas_iniciales = np.full(K, 0.5) 
    radios_iniciales = np.full(K, max(alto, ancho) * 0.05) # Radio más pequeño por alta densidad
    
    return LotePinceladasDiferenciables(
        K=K,
        x_iniciales=x_iniciales,
        y_iniciales=y_iniciales,
        angulos_iniciales=angulos_iniciales,
        alfas_iniciales=alfas_iniciales,
        colores_rgb_iniciales=np.array(colores_rgb_iniciales),
        radios_iniciales=radios_iniciales,
        tipo_figura="elipse"  # Cambiar a "gaussiana", "capsula", "rectangulo"
    )

def orquestar_reconstruccion():
    CONFIG = {
        "ruta_imagen": "data/proyecto.jpg",
        "directorio_salida": "data/ejemplo_simple",
        "alto_resolucion": 400,
        "ancho_resolucion": 500,
        "K_batch_size": 40,         
        "total_strokes": 1000,      
        "iteraciones_por_lote": 40,
        "learning_rate": 0.05,
        "w_mse": 0.8,
        "w_ssim": 1.2,
        "w_cantidad": 0.001,
        "w_area": 0.0010
    }

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["directorio_salida"], exist_ok=True)

    imagen_objetivo = cargar_imagen_objetivo(CONFIG["ruta_imagen"], CONFIG["alto_resolucion"], CONFIG["ancho_resolucion"], dispositivo)
    lienzo_actual = torch.zeros_like(imagen_objetivo).to(dispositivo)

    funcion_perdida = PerdidaCompuesta(
        peso_mse=CONFIG["w_mse"], peso_ssim=CONFIG["w_ssim"],
        peso_cantidad=CONFIG["w_cantidad"], peso_area=CONFIG["w_area"]
    )
    
    motor_opt = OptimizadorStrokes(imagen_objetivo, funcion_perdida, dispositivo)

    # El bucle ahora avanza en saltos de tamaño K
    total_lotes = CONFIG["total_strokes"] // CONFIG["K_batch_size"]
    pinceladas_acumuladas = 0
    
    for lote_idx in range(1, total_lotes + 1):
        lote_candidato = generar_lote_candidato(
            imagen_objetivo, lienzo_actual, 
            CONFIG["alto_resolucion"], CONFIG["ancho_resolucion"], CONFIG["K_batch_size"]
        )
        
        pinceladas_acumuladas += CONFIG["K_batch_size"]
        
        _, lienzo_actual, historial = motor_opt.optimizar_lote(
            lote_pinceladas=lote_candidato,
            lienzo_actual=lienzo_actual,
            cantidad_pinceladas=pinceladas_acumuladas,
            iteraciones=CONFIG["iteraciones_por_lote"],
            lr=CONFIG["learning_rate"]
        )
        
        if pinceladas_acumuladas == CONFIG["K_batch_size"] or pinceladas_acumuladas % 100 == 0:
            print(f"Progreso: [{pinceladas_acumuladas}/{CONFIG['total_strokes']}] strokes | MSE: {historial[-1]:.4f}")
            ruta_salida = os.path.join(CONFIG["directorio_salida"], f"reconstruccion_{pinceladas_acumuladas}.png")
            save_image(lienzo_actual, ruta_salida)

if __name__ == "__main__":
    orquestar_reconstruccion()
