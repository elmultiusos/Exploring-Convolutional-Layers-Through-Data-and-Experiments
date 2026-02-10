# UC Merced Land Use Classification with CNN + Swagger API

Sistema completo de clasificación de uso de tierra usando CNNs con deployment local mediante FastAPI y Swagger UI.

## 🌍 Descripción del Proyecto

Clasificación automática de imágenes satelitales en 21 categorías de uso de tierra usando Redes Neuronales Convolucionales (CNN).

### Dataset: UC Merced Land Use

- **Fuente**: USGS National Map Urban Area Imagery
- **Imágenes**: 2,100 imágenes RGB de 256×256 píxeles
- **Clases**: 21 categorías de uso de tierra
- **Resolución**: 1 pie por píxel
- **Distribución**: 100 imágenes por clase

### 21 Categorías

agricultural, airplane, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse, harbor, intersection, mediumresidential, mobilehomepark, overpass, parkinglot, river, runway, sparseresidential, storagetanks, tenniscourt

---

## 📁 Estructura del Proyecto

```
ucmerced-deployment/
├── save_model_ucmerced.py       # Script de entrenamiento
├── api_ucmerced.py              # API REST con Swagger
├── ucmerced_experiments.ipynb   # Notebook con análisis completo
│
├── label_map.json               # Mapeo de clases
├── train.csv                    # Metadata de entrenamiento
├── validation.csv               # Metadata de validación
├── test.csv                     # Metadata de prueba
│
├── requirements-windows.txt     # Dependencias compatibles Windows
├── install-windows.bat          # Instalador automático
├── TROUBLESHOOTING-WINDOWS.txt  # Guía de solución de problemas
│
├── models/                      # Modelos entrenados
│   ├── ucmerced_cnn.h5         # Modelo principal
│   └── ucmerced_cnn_classes.json
│
└── images_train_test_val/      # Dataset (debes tenerlo localmente)
    ├── train/
    ├── validation/
    └── test/
```

---

## 🚀 Inicio Rápido

### Prerequisitos

- Python 3.9-3.12 (NO Python 3.13)
- Carpeta `images_train_test_val` con el dataset organizado
- 8GB RAM recomendado
- GPU opcional (para entrenamiento más rápido)

### Instalación - WINDOWS

**Opción 1: Instalador Automático (Recomendado)**

```cmd
install-windows.bat
```

**Opción 2: Manual**

```cmd
# 1. Crear ambiente virtual
python -m venv venv
venv\Scripts\activate

# 2. Instalar TensorFlow y dependencias
pip install tensorflow
pip install fastapi uvicorn[standard] python-multipart
pip install numpy Pillow requests matplotlib pandas

# 3. Verificar instalación
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 🎓 Entrenamiento del Modelo

### Estructura de Datos Requerida

Asegúrate de tener esta estructura:

```
images_train_test_val/
├── train/
│   ├── agricultural/
│   ├── airplane/
│   ├── baseballdiamond/
│   └── ... (21 carpetas)
├── validation/
│   ├── agricultural/
│   └── ... (21 carpetas)
└── test/
    ├── agricultural/
    └── ... (21 carpetas)
```

### Entrenar el Modelo

```cmd
python save_model_ucmerced.py
```

**O con ruta personalizada:**

```cmd
python save_model_ucmerced.py path/to/images_train_test_val
```

### Configuración del Entrenamiento

```python
# En save_model_ucmerced.py:
IMG_HEIGHT = 128      # Tamaño de imagen (reduce de 256 para velocidad)
IMG_WIDTH = 128
BATCH_SIZE = 32       # Ajusta según tu RAM
EPOCHS = 20           # Número de epochs
```

**Salida esperada:**

```
🌍 UC MERCED LAND USE CLASSIFICATION - CNN Training
============================================================
Configuración:
  • Imagen: 128x128 RGB
  • Kernel size: (3, 3)
  • Batch size: 32
  • Epochs: 20
  • Clases: 21

📂 Cargando datos...
✓ Train samples: 1470
✓ Validation samples: 315
✓ Test samples: 315

🚀 Entrenando modelo por 20 epochs...
============================================================
Epoch 1/20
46/46 [==============================] - 45s - loss: 2.8145 - accuracy: 0.1524
...
✓ Resultados finales:
  Test Accuracy: 0.7968 (79.68%)
  Test Top-3 Accuracy: 0.9302 (93.02%)

✅ ¡Entrenamiento completado exitosamente!
```

---

## 🌐 Iniciar la API

```cmd
python api_ucmerced.py
```

**Salida:**

```
🚀 Iniciando UC Merced Land Use Classification API
📚 Documentación disponible en: http://localhost:8000/docs
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Acceder a Swagger UI

Abre tu navegador en: **http://localhost:8000/docs**

---

## 📊 Endpoints de la API

### Documentación Interactiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Disponibles

| Método | Endpoint         | Descripción                |
| ------ | ---------------- | -------------------------- |
| GET    | `/`              | Información básica         |
| GET    | `/health`        | Estado de la API           |
| GET    | `/model/info`    | Detalles del modelo        |
| GET    | `/classes`       | Lista de 21 clases         |
| GET    | `/examples`      | Ejemplos con descripciones |
| POST   | `/predict`       | **Clasificar imagen**      |
| POST   | `/batch/predict` | Clasificación en batch     |

### Ejemplo de Uso

**Python:**

```python
import requests

# Clasificar una imagen
with open('mi_imagen_satelital.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()

print(f"Clase predicha: {result['predicted_class']}")
print(f"Confianza: {result['confidence']:.2%}")
print(f"\nTop 3 predicciones:")
for pred in result['top_3_predictions']:
    print(f"  {pred['class']}: {pred['confidence']:.2%}")
```

**cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@imagen.jpg"
```

**Respuesta JSON:**

```json
{
  "predicted_class": "forest",
  "class_id": 7,
  "confidence": 0.8542,
  "top_3_predictions": [
    { "class": "forest", "confidence": 0.8542 },
    { "class": "chaparral", "confidence": 0.0892 },
    { "class": "agricultural", "confidence": 0.0234 }
  ]
}
```

---

## 🧪 Testing

### Script de Pruebas Manual

```python
# test_ucmerced_api.py
import requests
from PIL import Image
import numpy as np

# Probar con imagen generada
img = Image.new('RGB', (256, 256), color='green')
img.save('test_forest.png')

with open('test_forest.png', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', files={'file': f})
    print(response.json())
```

---

## 🏗️ Arquitectura del Modelo

### Diseño CNN

```python
Input (128, 128, 3)
    ↓
[Conv2D (32) + BN + MaxPool + Dropout(0.25)] ×1
    ↓
[Conv2D (64) + BN + MaxPool + Dropout(0.25)] ×1
    ↓
[Conv2D (128) + BN + MaxPool + Dropout(0.25)] ×1
    ↓
[Conv2D (256) + BN + MaxPool + Dropout(0.25)] ×1
    ↓
Flatten
    ↓
Dense(512) + Dropout(0.5)
    ↓
Dense(256) + Dropout(0.5)
    ↓
Dense(21, softmax)
```

### Características Clave

✅ **4 Bloques Convolucionales**: Extracción jerárquica de features  
✅ **Batch Normalization**: Estabiliza entrenamiento  
✅ **Dropout**: Previene overfitting (0.25 conv, 0.5 dense)  
✅ **Data Augmentation**: Rotación, zoom, flip horizontal  
✅ **Adam Optimizer**: Learning rate 0.001  
✅ **Early Stopping**: Paciencia de 5 epochs

### Métricas

- **Accuracy**: Predicción exacta
- **Top-3 Accuracy**: Clase correcta en top 3 predicciones

---

## 📈 Resultados Esperados

### Baseline vs CNN

| Modelo    | Arquitectura  | Accuracy   | Top-3      | Parámetros |
| --------- | ------------- | ---------- | ---------- | ---------- |
| Baseline  | Dense only    | ~45-55%    | N/A        | ~2.5M      |
| CNN (3×3) | 4 Conv blocks | **75-85%** | **93-96%** | ~1.8M      |
| CNN (5×5) | 4 Conv blocks | 72-82%     | 91-95%     | ~2.1M      |

### Análisis por Clase

**Fáciles de clasificar:**

- airplane (>90%)
- baseballdiamond (>90%)
- tenniscourt (>90%)
- runway (>88%)

**Difíciles de clasificar:**

- denseresidential vs mediumresidential (~70%)
- buildings vs denseresidential (~72%)
- chaparral vs forest (~75%)

---

## 🔬 Experimentos del Notebook

El notebook `ucmerced_experiments.ipynb` contiene:

1. **EDA completo**: Distribución de clases, visualización
2. **Modelo baseline**: MLP con capas densas
3. **Arquitectura CNN**: Diseño justificado
4. **Experimentos controlados**: 3×3 vs 5×5 kernels
5. **Interpretación**: Por qué CNNs funcionan mejor
6. **Visualizaciones**: Curvas de aprendizaje, métricas

### Ejecutar el Notebook

```cmd
# Instalar Jupyter
pip install jupyter

# Iniciar
jupyter notebook ucmerced_experiments.ipynb
```

---

## 🛠️ Configuración Avanzada

### Cambiar Tamaño de Imagen

```python
# En save_model_ucmerced.py:
IMG_HEIGHT = 256  # Original (más lento pero mejor accuracy)
IMG_WIDTH = 256
```

### Ajustar Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Aumentar rotación
    width_shift_range=0.3,  # Más desplazamiento
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,     # Agregar flip vertical
    zoom_range=0.3
)
```

### Transfer Learning (Opcional)

```python
# Usar modelo pre-entrenado
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)
```

---

## ❓ Troubleshooting

### Error: "No module named 'tensorflow'"

```cmd
pip install tensorflow
```

### Error: Directorio no encontrado

```
❌ Error: No se encuentra el directorio 'images_train_test_val'
```

**Solución**: Asegúrate de tener la estructura correcta:

```cmd
dir images_train_test_val\train
# Debe mostrar 21 carpetas
```

### Error: Out of Memory

**Reduce batch size:**

```python
BATCH_SIZE = 16  # En vez de 32
```

**O reduce tamaño de imagen:**

```python
IMG_HEIGHT = 64
IMG_WIDTH = 64
```

### Modelo predice siempre la misma clase

- Verifica que el dataset esté balanceado
- Aumenta epochs (mínimo 15-20)
- Reduce learning rate a 0.0005
- Verifica data augmentation

### Consulta la guía completa

Ver **TROUBLESHOOTING-WINDOWS.txt** para más soluciones.

## 📚 Referencias

**Paper Original:**
Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.

**Dataset:**
http://weegee.vision.ucmerced.edu/datasets/landuse.html

**Contacto Dataset:**
Shawn D. Newsam  
University of California, Merced  
snewsam@ucmerced.edu

---

## 📄 Licencia

Este proyecto es para propósitos educativos. El dataset UC Merced es de dominio público (USGS imagery).

## Author

Juan Sebastian Buitrago Piñeros
Universidad Escuela Colombiana de Ingenieria Julio Garavito

---
