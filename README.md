# Reconstrucción Progresiva de Imágenes con Pinceladas Diferenciables

Este repositorio contiene un motor de optimización iterativa implementado en PyTorch. El sistema aproxima imágenes objetivo mediante la superposición de primitivas geométricas continuas (elipses, rectángulos, círculos, gaussianas y cápsulas), optimizando sus parámetros (posición, color, escala, rotación y opacidad) a través de descenso de gradiente.

## Características Principales

* **Procesamiento por Lotes (Batching):** Optimización simultánea de $K$ pinceladas para maximizar la saturación de núcleos CUDA.
* **Rasterización Analítica (SDF):** Modelado de límites geométricos mediante Campos de Distancia con Signo diferenciables.
* **Función de Pérdida Compuesta:** Integración de Error Cuadrático Medio (MSE), Similitud Estructural (SSIM), y Pérdida Perceptual semántica (VGG16).
* **Aceleración por Hardware:** Uso de Precisión Mixta Automática (AMP) para reducir a la mitad el consumo de VRAM y acelerar el cálculo matricial.
* **Heurística Multinomial:** Inicialización inteligente de trazos guiada por mapas de error residual.

## Estructura del Directorio

```text
/
├── data/
│   ├── proyecto.jpg                 # Imagen objetivo
│   └── ejemplo_simple/              # Directorio de salida (imágenes y GIF)
├── src/
│   ├── main.py                      # Orquestador y bucle principal
│   ├── motor_optimizacion.py        # Algoritmo de actualización (Adam + AMP)
│   ├── perdida_compuesta.py         # Función objetivo (MSE + SSIM + VGG16 + Regularización)
│   ├── metricas_error.py            # Implementaciones base de errores matemáticos
│   ├── pinceladas_lote.py           # Definición matemática de primitivas y Alpha Compositing
│   └── generar_gif.py               # Compilador de secuencias de frames a GIF animado
└── requirements.txt                 # Dependencias del proyecto

```

## Requisitos e Instalación

El proyecto requiere una GPU compatible con CUDA para un rendimiento óptimo, especialmente debido a la extracción de características con VGG16.

1. Clona el repositorio.
2. Instala las dependencias en tu entorno virtual:

```bash
pip install torch torchvision numpy Pillow

```

## Uso

### 1. Ejecución de la Optimización

El orquestador principal inicializa el lienzo y procesa iterativamente los lotes de pinceladas. Modifica el diccionario `CONFIG` dentro de `src/main.py` para ajustar hiperparámetros (tamaño de lote, pesos de pérdida, tipo de figura).

Para iniciar la reconstrucción:

```bash
python -m src.main

```

Los fotogramas intermedios se guardarán en el directorio `data/ejemplo_simple/` por defecto en cada iteración de lote.

### 2. Generación de Animación

Una vez que el algoritmo finalice (o se interrumpa con suficientes fotogramas generados), compila la progresión en un archivo GIF:

```bash
python -m src.generar_gif

```

El archivo resultante (`evolucion_strokes.gif`) mostrará la transición de la imagen desde las pinceladas expansivas de baja frecuencia hasta los detalles de alta frecuencia.

## Ajuste de la Función Objetivo

El balance de la reconstrucción se controla desde `main.py` ajustando los siguientes pesos en la configuración:

* `w_mse`: Fuerza la coincidencia estricta de color por píxel.
* `w_ssim`: Mejora la percepción de contraste y textura local.
* `w_vgg`: Extrae mapas de características profundas para igualar bordes y semántica espacial (crítico para detalles).
* `w_area`: Penalización geométrica; valores altos obligan al modelo a utilizar trazos pequeños para texturas finas.
