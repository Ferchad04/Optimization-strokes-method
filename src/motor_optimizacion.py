import torch
import torch.optim as optim

class OptimizadorStrokes:
    """
    Controla el bucle de optimización para ajustar los parámetros de una pincelada
    minimizando la pérdida respecto a la imagen objetivo.
    """
    def __init__(self, imagen_objetivo, metrica_perdida, dispositivo=None):
        # Configuración de hardware para procesamiento tensorial
        if dispositivo is None:
            self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dispositivo = dispositivo
            
        self.imagen_objetivo = imagen_objetivo.to(self.dispositivo)
        self.metrica_perdida = metrica_perdida.to(self.dispositivo)

    def optimizar_stroke(self, stroke, lienzo_actual, iteraciones=100, metodo='adam', lr=0.01):
        """
        Ejecuta el ciclo de actualización de gradientes para un stroke específico.
        
        Args:
            stroke (nn.Module): Instancia de PinceladaDiferenciable.
            lienzo_actual (Tensor): Estado actual de la reconstrucción.
            iteraciones (int): Número de pasos de optimización.
            metodo (str): 'adam' (Gradiente Descendente) o 'lbfgs' (Quasi-Newton).
            lr (float): Tasa de aprendizaje.
        """
        stroke = stroke.to(self.dispositivo)
        lienzo_actual = lienzo_actual.to(self.dispositivo)
        alto, ancho = self.imagen_objetivo.shape[1], self.imagen_objetivo.shape[2]

        # Selección del algoritmo de optimización
        if metodo == 'adam':
            optimizador = optim.Adam(stroke.parameters(), lr=lr)
        elif metodo == 'lbfgs':
            # L-BFGS requiere un closure para reevaluar la función objetivo múltiples veces
            optimizador = optim.LBFGS(stroke.parameters(), lr=lr)
        else:
            raise ValueError("Método no soportado. Use 'adam' o 'lbfgs'.")

        historial_perdida = []

        for i in range(iteraciones):
            if metodo == 'adam':
                optimizador.zero_grad()
                
                # Forward pass: renderizar y superponer
                stroke_renderizado, mascara = stroke.renderizar_en_lienzo(alto, ancho)
                lienzo_propuesto = lienzo_actual * (1 - mascara) + stroke_renderizado
                
                # Calcular error
                perdida = self.metrica_perdida.mse(self.imagen_objetivo, lienzo_propuesto)
                
                # Backward pass
                perdida.backward()
                optimizador.step()
                
                historial_perdida.append(perdida.item())
                
            elif metodo == 'lbfgs':
                def closure():
                    optimizador.zero_grad()
                    stroke_renderizado, mascara = stroke.renderizar_en_lienzo(alto, ancho)
                    lienzo_propuesto = lienzo_actual * (1 - mascara) + stroke_renderizado
                    perdida = self.metrica_perdida.mse(self.imagen_objetivo, lienzo_propuesto)
                    perdida.backward()
                    return perdida
                
                optimizador.step(closure)
                # L-BFGS ejecuta múltiples pasos internos, se registra la pérdida final del paso
                with torch.no_grad():
                    stroke_renderizado, mascara = stroke.renderizar_en_lienzo(alto, ancho)
                    lienzo_propuesto = lienzo_actual * (1 - mascara) + stroke_renderizado
                    perdida_actual = self.metrica_perdida.mse(self.imagen_objetivo, lienzo_propuesto)
                    historial_perdida.append(perdida_actual.item())

            # Restricciones de parámetros (ej. mantener colores entre 0 y 1)
            with torch.no_grad():
                stroke.color_rgb.clamp_(0.0, 1.0)
                stroke.transparencia_alfa.clamp_(0.0, 1.0)
                stroke.tamano_radio.clamp_(min=1.0) # Evitar radios negativos o cero

        # Devolver el stroke optimizado y la pérdida final
        stroke_renderizado_final, mascara_final = stroke.renderizar_en_lienzo(alto, ancho)
        lienzo_final = lienzo_actual * (1 - mascara_final) + stroke_renderizado_final
        
        return stroke, lienzo_final, historial_perdida