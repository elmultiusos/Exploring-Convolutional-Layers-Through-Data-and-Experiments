# Exploring Convolutional Layers: Fashion-MNIST Experiments

Este repositorio contiene la solución completa para el taller de "Exploring Convolutional Layers Through Data and Experiments".

## Estructura del Proyecto

- `convolutional_experiments.ipynb`: Notebook principal con todo el código, experimentos y análisis.
- `train.py`: Script de entrenamiento para despliegue en SageMaker (generado dentro del notebook).
- `README.md`: Este documento.

## Descripción del Problema
El objetivo es analizar cómo las capas convolucionales introducen un "inductive bias" adecuado para datos de imágenes, comparando su desempeño contra modelos densos (Fully Connected) y experimentando con hiperparámetros arquitecturales.

## Dataset
**Fashion-MNIST**
- 60,000 imágenes de entrenamiento, 10,000 de prueba.
- 10 clases de ropa/calzado (28x28 grayscale).
- Preprocesamiento: Normalización (0-1) y reshape a (N, 28, 28, 1).

## Arquitectura CNN
Se diseñó una arquitectura CNN simple pero efectiva:
1. **Conv2D** (32 filtros, ReLU)
2. **MaxPooling2D** (2x2)
3. **Conv2D** (64 filtros, ReLU)
4. **MaxPooling2D** (2x2)
5. **Flatten**
6. **Dense** (64 unidades, ReLU)
7. **Output Dense** (10 unidades, Softmax)

## Experimentos y Resultados
Se comparó el efecto del **Kernel Size** (3x3 vs 5x5).

| Modelo | Kernel Size | Accuracy (Val) | Observaciones |
|--------|-------------|----------------|---------------|
| Baseline (Dense) | N/A | ~88% | Pierde estructura espacial. |
| CNN A | 3x3 | ~91% | Mejor captura de detalles finos/texturas. |
| CNN B | 5x5 | ~90% | Receptive field más amplio, ligeramente menos preciso en este dataset. |

**Conclusión Principal**: Las CNNs superan consistentemente al modelo base denso debido a su capacidad de capturar invarianza traslacional y jerarquías espaciales con menos parámetros efectivos.

## Despliegue en SageMaker
El notebook incluye una sección final con el código necesario para entrenar y desplegar este modelo utilizando el SDK de `sagemaker`. Se utiliza un `TensorFlow Estimator` para ejecutar el script `train.py` en una instancia `ml.m5.xlarge`.

## Cómo ejecutar
1. Instalar dependencias: `pip install tensorflow numpy matplotlib sagemaker`
2. Abrir Jupyter: `jupyter notebook`
3. Ejecutar `convolutional_experiments.ipynb` secuencialmente.
