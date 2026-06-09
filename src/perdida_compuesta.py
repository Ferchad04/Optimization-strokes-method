import torch
import torch.nn as nn
import math

class PerdidaCompuesta(nn.Module):
    """
    Combina fidelidad visual (MSE, SSIM) y regularización geométrica 
    """
    def __init__(self, peso_mse=1.0, peso_ssim=1.0, peso_cantidad=0.01, peso_area=0.005):
        super(PerdidaCompuesta, self).__init__()

        self.alfa = peso_mse
        self.beta = peso_ssim
        self.gamma = peso_cantidad
        self.lambda_reg = peso_area
        
        from src.metricas_error import MetricasReconstruccion
        self.metricas_base = MetricasReconstruccion(tipo_reduccion='mean')

    def forward(self, imagen_original, lienzo_propuesto, cantidad_pinceladas, stroke_actual):
        """
        Calcula la función objetivo compuesta con penalizaciones.
        """
        # Errores de reconstrucción visual
        perdida_mse = self.metricas_base.mse(imagen_original, lienzo_propuesto)
        perdida_ssim = self.metricas_base.perdida_ssim(imagen_original, lienzo_propuesto)
        
        # Penalizaciones de regularización 
        # Penalización por número de strokes: L = Error + gamma * N_strokes
        penalizacion_cantidad = torch.tensor(float(cantidad_pinceladas), device=imagen_original.device)
        
        # Penalización por tamaño: L = Error + lambda * area(s)
        # Penalización por tamaño vectorizada: L = Error + lambda * sum(area(s_k))
        if stroke_actual.tipo_figura == "circulo":
            area_estimada = torch.sum(torch.pi * (stroke_actual.tamano_radio ** 2))
        elif stroke_actual.tipo_figura == "rectangulo":
            area_estimada = torch.sum((stroke_actual.tamano_radio * 2) ** 2)
        else:
            area_estimada = torch.tensor(0.0, device=imagen_original.device)
            
        perdida_total = (self.alfa * perdida_mse) + \
                        (self.beta * perdida_ssim) + \
                        (self.gamma * penalizacion_cantidad) + \
                        (self.lambda_reg * area_estimada)
        
        desglose_monitoreo = {
            "perdida_total": perdida_total,
            "mse": perdida_mse.item(),
            "ssim": perdida_ssim.item(),
            "reg_cantidad": penalizacion_cantidad.item(),
            "reg_area": area_estimada.item()
        }
        
        return perdida_total, desglose_monitoreo