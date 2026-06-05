import torch
import torch.nn as nn

class MetricasReconstruccion(nn.Module):
    """
    Calcula el error entre la imagen objetivo (I) y la aproximación generada (I_hat).
    Hereda de nn.Module para su fácil integración en pipelines de optimización.
    """
    def __init__(self, tipo_reduccion='sum'):
        """
        Args:
            tipo_reduccion (str): 'sum' aplica la sumatoria estricta según las ecuaciones.
                                  'mean' aplica el promedio (útil para estabilidad de gradientes).
        """
        super(MetricasReconstruccion, self).__init__()
        self.tipo_reduccion = tipo_reduccion

    def mse(self, I, I_hat):
        """
        Calcula el Error Cuadrático Medio.
        """
        diferencia_cuadratica = (I - I_hat) ** 2
        
        if self.tipo_reduccion == 'sum':
            return torch.sum(diferencia_cuadratica)
        return torch.mean(diferencia_cuadratica)

    def mae(self, I, I_hat):
        """
        Calcula el Error Absoluto Medio.
        """
        diferencia_absoluta = torch.abs(I - I_hat)
        
        if self.tipo_reduccion == 'sum':
            return torch.sum(diferencia_absoluta)
        return torch.mean(diferencia_absoluta)