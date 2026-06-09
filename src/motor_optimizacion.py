import torch
import torch.optim as optim

class OptimizadorStrokes:
    """
    Controla el bucle de optimización para ajustar los parámetros de un lote 
    de K pinceladas minimizando la pérdida global.
    """
    def __init__(self, imagen_objetivo, metrica_perdida, dispositivo=None):
        if dispositivo is None:
            self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dispositivo = dispositivo
            
        self.imagen_objetivo = imagen_objetivo.to(self.dispositivo)
        self.metrica_perdida = metrica_perdida.to(self.dispositivo)

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
            
            # Forward pass vectorial
            lote_coloreado, lote_mascaras = lote_pinceladas.renderizar_lote(alto, ancho)
            lienzo_propuesto = lote_pinceladas.componer_lienzo(lote_coloreado, lote_mascaras, lienzo_actual)
            
            # Cálculo de error global frente a la suma del lote
            perdida, _ = self.metrica_perdida(self.imagen_objetivo, lienzo_propuesto, cantidad_pinceladas, lote_pinceladas)
            
            # Backward pass 
            perdida.backward()
            optimizador.step()
            historial_perdida.append(perdida.item())
                
            # Restricciones post-actualización
            with torch.no_grad():
                lote_pinceladas.color_rgb.clamp_(0.0, 1.0)
                lote_pinceladas.transparencia_alfa.clamp_(0.0, 1.0)
                lote_pinceladas.tamano_radio.clamp_(min=1.0)

        # Renderizado final con el grafo desconectado
        lote_coloreado_final, lote_mascaras_final = lote_pinceladas.renderizar_lote(alto, ancho)
        lienzo_final = lote_pinceladas.componer_lienzo(lote_coloreado_final, lote_mascaras_final, lienzo_actual)
        
        return lote_pinceladas, lienzo_final.detach(), historial_perdida