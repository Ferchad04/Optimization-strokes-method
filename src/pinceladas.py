import torch
import torch.nn as nn

class PinceladaDiferenciable(nn.Module):
    """
    Representa una sola pincelada (stroke) como una figura geométrica 
    con parámetros optimizables.
    """
    def __init__(self, x_inicial, y_inicial, angulo_inicial, alfa_inicial, color_rgb_inicial, radio_inicial, tipo_figura="circulo"):
        super(PinceladaDiferenciable, self).__init__()
        
        self.tipo_figura = tipo_figura
        
        # Envolvemos los valores en nn.Parameter para indicar que son las 
        # variables que el motor de optimización va a ajustar iterativamente.
        self.posicion_x = nn.Parameter(torch.tensor([x_inicial], dtype=torch.float32))
        self.posicion_y = nn.Parameter(torch.tensor([y_inicial], dtype=torch.float32))
        self.angulo_theta = nn.Parameter(torch.tensor([angulo_inicial], dtype=torch.float32))
        self.transparencia_alfa = nn.Parameter(torch.tensor([alfa_inicial], dtype=torch.float32))
        
        # El color es un tensor de 3 elementos para los canales R, G, B
        self.color_rgb = nn.Parameter(torch.tensor(color_rgb_inicial, dtype=torch.float32))
        
        # Añadimos un parámetro de tamaño (radio para círculo, escala para rectángulo)
        self.tamano_radio = nn.Parameter(torch.tensor([radio_inicial], dtype=torch.float32))

    def renderizar_en_lienzo(self, alto_lienzo, ancho_lienzo):
        """
        Genera una matriz del tamaño del lienzo con la figura dibujada.
        Se utilizan operaciones matemáticas continuas para no romper el cálculo de gradientes.
        """
        # 1. Crear una cuadrícula de coordenadas (X, Y) para todos los píxeles del lienzo
        vector_y = torch.arange(0, alto_lienzo, dtype=torch.float32)
        vector_x = torch.arange(0, ancho_lienzo, dtype=torch.float32)
        coordenadas_y, coordenadas_x = torch.meshgrid(vector_y, vector_x, indexing='ij')
        
        # 2. Calcular la distancia de cada píxel al centro de la pincelada
        distancia_x = coordenadas_x - self.posicion_x
        distancia_y = coordenadas_y - self.posicion_y
        
        # 3. Dibujar la figura basándonos en la distancia matemática
        if self.tipo_figura == "circulo":
            # Ecuación del círculo: x^2 + y^2 = r^2
            distancia_al_cuadrado = (distancia_x ** 2) + (distancia_y ** 2)
            limite_figura = (self.tamano_radio ** 2) - distancia_al_cuadrado
            
        elif self.tipo_figura == "rectangulo":
            # Para el rectángulo aplicamos la rotación (ángulo_theta)
            coseno_angulo = torch.cos(self.angulo_theta)
            seno_angulo = torch.sin(self.angulo_theta)
            
            x_rotado = distancia_x * coseno_angulo - distancia_y * seno_angulo
            y_rotado = distancia_x * seno_angulo + distancia_y * coseno_angulo
            
            # Verificamos si los puntos caen dentro del área del rectángulo
            distancia_maxima = torch.maximum(torch.abs(x_rotado), torch.abs(y_rotado))
            limite_figura = self.tamano_radio - distancia_maxima
        else:
            raise ValueError("Tipo de figura no soportado. Usa 'circulo' o 'rectangulo'.")

        # 4. Rasterización analítica (Suavizado matemático)
        # Usamos la función sigmoide para que el borde de la figura sea continuo y diferenciable.
        multiplicador_nitidez = 5.0 
        mascara_figura = torch.sigmoid(limite_figura * multiplicador_nitidez)
        
        # 5. Aplicar la transparencia (alfa) a la figura
        mascara_con_transparencia = mascara_figura * torch.sigmoid(self.transparencia_alfa)
        
        # 6. Colorear la figura (Multiplicar la máscara por los canales RGB)
        # Reestructuramos el tensor de color para poder multiplicarlo por la matriz 2D
        color_reestructurado = self.color_rgb.view(3, 1, 1) 
        lienzo_coloreado = color_reestructurado * mascara_con_transparencia
        
        return lienzo_coloreado, mascara_con_transparencia