import torch
import torch.nn as nn

class LotePinceladasDiferenciables(nn.Module):
    def __init__(self, K, x_iniciales, y_iniciales, angulos_iniciales, alfas_iniciales, colores_rgb_iniciales, radios_iniciales, tipo_figura="circulo"):
        super(LotePinceladasDiferenciables, self).__init__()
        
        self.K = K
        self.tipo_figura = tipo_figura
        
        # Parámetros espaciales base [K, 1, 1]
        self.posicion_x = nn.Parameter(torch.tensor(x_iniciales, dtype=torch.float32).view(K, 1, 1))
        self.posicion_y = nn.Parameter(torch.tensor(y_iniciales, dtype=torch.float32).view(K, 1, 1))
        self.angulo_theta = nn.Parameter(torch.tensor(angulos_iniciales, dtype=torch.float32).view(K, 1, 1))
        self.transparencia_alfa = nn.Parameter(torch.tensor(alfas_iniciales, dtype=torch.float32).view(K, 1, 1))
        self.tamano_radio = nn.Parameter(torch.tensor(radios_iniciales, dtype=torch.float32).view(K, 1, 1))
        
        # Parámetros adicionales para Elipses, Rectángulos y Cápsulas
        self.relacion_aspecto = nn.Parameter(torch.ones(K, 1, 1, dtype=torch.float32))
        self.longitud = nn.Parameter(torch.tensor(radios_iniciales, dtype=torch.float32).view(K, 1, 1) * 2.0)
        
        # Parámetro de color [K, 3, 1, 1]
        self.color_rgb = nn.Parameter(torch.tensor(colores_rgb_iniciales, dtype=torch.float32).view(K, 3, 1, 1))

    def renderizar_lote(self, alto_lienzo, ancho_lienzo):
        vector_y = torch.arange(0, alto_lienzo, dtype=torch.float32, device=self.posicion_x.device)
        vector_x = torch.arange(0, ancho_lienzo, dtype=torch.float32, device=self.posicion_x.device)
        coordenadas_y, coordenadas_x = torch.meshgrid(vector_y, vector_x, indexing='ij')
        
        coordenadas_y = coordenadas_y.unsqueeze(0)
        coordenadas_x = coordenadas_x.unsqueeze(0)
        
        distancia_x = coordenadas_x - self.posicion_x
        distancia_y = coordenadas_y - self.posicion_y
        
        # Pre-calcular rotaciones para figuras direccionales
        coseno_angulo = torch.cos(self.angulo_theta)
        seno_angulo = torch.sin(self.angulo_theta)
        x_rotado = distancia_x * coseno_angulo - distancia_y * seno_angulo
        y_rotado = distancia_x * seno_angulo + distancia_y * coseno_angulo

        multiplicador_nitidez = 5.0 
        
        # 1. CÍRCULO
        if self.tipo_figura == "circulo":
            distancia_al_cuadrado = (distancia_x ** 2) + (distancia_y ** 2)
            limite_figura = (self.tamano_radio ** 2) - distancia_al_cuadrado
            mascara_figura = torch.sigmoid(limite_figura * multiplicador_nitidez)
            
        # 2. RECTÁNGULO (Modificado con relación de aspecto)
        elif self.tipo_figura == "rectangulo":
            distancia_maxima = torch.maximum(torch.abs(x_rotado), torch.abs(y_rotado) * self.relacion_aspecto)
            limite_figura = self.tamano_radio - distancia_maxima
            mascara_figura = torch.sigmoid(limite_figura * multiplicador_nitidez)
            
        # 3. ELIPSE
        elif self.tipo_figura == "elipse":
            # Ecuación de la elipse normalizada para mantener gradientes estables
            dist_sq_x = (x_rotado / self.tamano_radio) ** 2
            dist_sq_y = (y_rotado / (self.tamano_radio * self.relacion_aspecto)) ** 2
            limite_figura = 1.0 - (dist_sq_x + dist_sq_y)
            mascara_figura = torch.sigmoid(limite_figura * multiplicador_nitidez)
            
        # 4. GAUSSIANA
        elif self.tipo_figura == "gaussiana":
            distancia_al_cuadrado = (distancia_x ** 2) + (distancia_y ** 2)
            mascara_figura = torch.exp(-distancia_al_cuadrado / (2 * (self.tamano_radio ** 2)))
            
        # 5. CÁPSULA (Cilindro con bordes redondeados)
        elif self.tipo_figura == "capsula":
            mitad_longitud = self.longitud / 2.0
            x_clamp = torch.clamp(x_rotado, -mitad_longitud, mitad_longitud)
            distancia_al_cuadrado = (x_rotado - x_clamp)**2 + (y_rotado)**2
            limite_figura = (self.tamano_radio ** 2) - distancia_al_cuadrado
            mascara_figura = torch.sigmoid(limite_figura * multiplicador_nitidez)
            
        else:
            raise ValueError(f"Figura '{self.tipo_figura}' no soportada.")

        # Aplicar transparencia alfa
        mascara_con_transparencia = mascara_figura * torch.sigmoid(self.transparencia_alfa)
        mascara_expandida = mascara_con_transparencia.unsqueeze(1)
        
        # Coloreado vectorizado
        lote_coloreado = self.color_rgb * mascara_expandida
        
        return lote_coloreado, mascara_expandida

    def componer_lienzo(self, lote_coloreado, lote_mascaras, lienzo_base):
        lienzo_actual = lienzo_base.clone()
        for k in range(self.K):
            lienzo_actual = lienzo_actual * (1 - lote_mascaras[k]) + lote_coloreado[k]
        return lienzo_actual
