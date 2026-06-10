import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler

class OptimizadorStrokes:
    """
    Controla el bucle de optimización para ajustar los parámetros de un lote 
    de K pinceladas minimizando la pérdida global mediante Precisión Mixta (AMP).
    """
    def __init__(self, imagen_objetivo, metrica_perdida, dispositivo=None):
        if dispositivo is None:
            self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dispositivo = dispositivo
            
        self.imagen_objetivo = imagen_objetivo.to(self.dispositivo)
        self.metrica_perdida = metrica_perdida.to(self.dispositivo)
        
        # Inicializador del escalador de gradientes para evitar el desbordamiento a cero en float16
        self.es_cuda = self.dispositivo.type == 'cuda'
        self.scaler = GradScaler(enabled=self.es_cuda)

    def optimizar_lote(self, lote_pinceladas, lienzo_actual, cantidad_pinceladas, iteraciones=50, lr=0.1):
        """
        Ejecuta la actualización de gradientes para un lote (Batch) completo.
        """
        lote_pinceladas = lote_pinceladas.to(self.dispositivo)
        lienzo_actual = lienzo_actual.to(self.dispositivo)
        alto, ancho = self.imagen_objetivo.shape[1], self.imagen_objetivo.shape[2]

        optimizador = optim.Adam(lote_pinceladas.parameters(), lr=lr)
        historial_perdida = []

        for _ in range(iteraciones):
            optimizador.zero_grad()
            
            # Contexto AMP: Ejecuta el forward pass en precisión media (float16)
            with autocast(device_type=self.dispositivo.type, enabled=self.es_cuda):
                lote_coloreado, lote_mascaras = lote_pinceladas.renderizar_lote(alto, ancho)
                lienzo_propuesto = lote_pinceladas.componer_lienzo(lote_coloreado, lote_mascaras, lienzo_actual)
                
                perdida, _ = self.metrica_perdida(self.imagen_objetivo, lienzo_propuesto, cantidad_pinceladas, lote_pinceladas)
            
            # Backward pass escalado para mantener precisión numérica
            self.scaler.scale(perdida).backward()
            self.scaler.step(optimizador)
            self.scaler.update()
            
            historial_perdida.append(perdida.item())
                
            # Restricciones post-actualización
            with torch.no_grad():
                lote_pinceladas.color_rgb.clamp_(0.0, 1.0)
                lote_pinceladas.transparencia_alfa.clamp_(0.0, 1.0)
                lote_pinceladas.tamano_radio.clamp_(min=1.0)
                # Restricciones de figuras secundarias si aplican
                if hasattr(lote_pinceladas, 'relacion_aspecto'):
                    lote_pinceladas.relacion_aspecto.clamp_(min=0.1)
                if hasattr(lote_pinceladas, 'longitud'):
                    lote_pinceladas.longitud.clamp_(min=1.0)

        # Renderizado final con el grafo desconectado
        # Se ejecuta fuera del contexto autocast para devolver float32 estándar al lienzo acumulado
        with torch.no_grad():
            lote_coloreado_final, lote_mascaras_final = lote_pinceladas.renderizar_lote(alto, ancho)
            lienzo_final = lote_pinceladas.componer_lienzo(lote_coloreado_final, lote_mascaras_final, lienzo_actual)
        
        return lote_pinceladas, lienzo_final.detach(), historial_perdida