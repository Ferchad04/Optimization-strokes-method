import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


    def distancia_rgb(self, I, I_hat):
        """
        Calcula la distancia al cuadrado en el espacio de color RGB.
        Matemáticamente similar al MSE, pero conceptualizada por canal de color.
        L = suma || c_ij - c_hat_ij ||^2
        """
        # Calculamos la norma al cuadrado a lo largo del canal de color (dimensión 0 si es [C, H, W])
        distancia_color = torch.sum((I - I_hat) ** 2, dim=0)
        
        if self.tipo_reduccion == 'sum':
            return torch.sum(distancia_color)
        return torch.mean(distancia_color)

    def perdida_ssim(self, I, I_hat, tamano_ventana=11, sigma=1.5):
        """
        Calcula la pérdida basada en el Índice de Similitud Estructural (SSIM).
        L = 1 - SSIM(I, I_hat)
        """

        # Creamos un kernel gaussiano 1D y expandimos a 2D
        coordenadas = torch.arange(tamano_ventana, dtype=torch.float32, device=I.device)
        coordenadas -= (tamano_ventana - 1) / 2.0
        gaussiana_1d = torch.exp(-(coordenadas ** 2) / (2 * sigma ** 2))
        gaussiana_1d = gaussiana_1d / torch.sum(gaussiana_1d)
        kernel_2d = gaussiana_1d.unsqueeze(0) * gaussiana_1d.unsqueeze(1)
        
        # Adaptamos el kernel para imágenes RGB (3 canales)
        canales = I.shape[0]
        kernel_2d = kernel_2d.expand(canales, 1, tamano_ventana, tamano_ventana)
        
        # Aseguramos que las imágenes tengan formato [Batch, Canal, Alto, Ancho] para la convolución
        I_batch = I.unsqueeze(0)
        I_hat_batch = I_hat.unsqueeze(0)

        # Calculamos las medias ponderadas locales (mu)
        mu1 = F.conv2d(I_batch, kernel_2d, padding=tamano_ventana//2, groups=canales)
        mu2 = F.conv2d(I_hat_batch, kernel_2d, padding=tamano_ventana//2, groups=canales)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Calculamos las varianzas y covarianzas locales (sigma)
        sigma1_sq = F.conv2d(I_batch * I_batch, kernel_2d, padding=tamano_ventana//2, groups=canales) - mu1_sq
        sigma2_sq = F.conv2d(I_hat_batch * I_hat_batch, kernel_2d, padding=tamano_ventana//2, groups=canales) - mu2_sq
        sigma12 = F.conv2d(I_batch * I_hat_batch, kernel_2d, padding=tamano_ventana//2, groups=canales) - mu1_mu2

        # Constantes de estabilidad (basadas en rango de píxeles 0-1)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Ecuación final de SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # El SSIM promedio de la imagen
        ssim_valor = ssim_map.mean()
        
        # Retornamos la pérdida 
        return 1.0 - ssim_valor