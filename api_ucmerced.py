"""
API REST para servir el modelo CNN de UC Merced Land Use Classification
Incluye documentación automática con Swagger UI
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from typing import List, Dict
import uvicorn
import json
import os

# Inicializar FastAPI
app = FastAPI(
    title="UC Merced Land Use Classification API",
    description="API para clasificación de uso de tierra usando Redes Neuronales Convolucionales. Dataset: UC Merced Land Use (21 clases)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Cargar mapeo de clases
CLASS_MAP_PATH = 'models/ucmerced_cnn_classes.json'
LABEL_MAP_PATH = 'label_map.json'

# Inicializar
CLASS_NAMES = {}
CLASS_INDICES = {}
model = None
MODEL_PATH = 'models/ucmerced_cnn.h5'
IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_class_mappings():
    """Carga el mapeo de clases desde los archivos JSON"""
    global CLASS_NAMES, CLASS_INDICES
    
    # Intentar cargar desde el archivo del modelo
    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, 'r') as f:
            CLASS_INDICES = json.load(f)
            # Invertir el mapeo
            CLASS_NAMES = {v: k for k, v in CLASS_INDICES.items()}
    elif os.path.exists(LABEL_MAP_PATH):
        # Usar label_map.json como fallback
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = json.load(f)
            CLASS_INDICES = label_map
            CLASS_NAMES = {v: k for k, v in label_map.items()}
    else:
        # Mapeo por defecto
        CLASS_NAMES = {
            0: "agricultural", 1: "airplane", 2: "baseballdiamond", 3: "beach",
            4: "buildings", 5: "chaparral", 6: "denseresidential", 7: "forest",
            8: "freeway", 9: "golfcourse", 10: "intersection", 11: "mediumresidential",
            12: "mobilehomepark", 13: "overpass", 14: "parkinglot", 15: "river",
            16: "runway", 17: "sparseresidential", 18: "storagetanks", 19: "tenniscourt",
            20: "harbor"
        }
        CLASS_INDICES = {v: k for k, v in CLASS_NAMES.items()}

@app.on_event("startup")
async def load_model():
    """Carga el modelo y mapeos al iniciar la API"""
    global model
    try:
        # Cargar mapeos de clases
        load_class_mappings()
        print(f"✓ Mapeos de clases cargados: {len(CLASS_NAMES)} clases")
        
        # Cargar modelo
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✓ Modelo cargado desde {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Error al cargar modelo: {e}")
        print("Asegúrate de ejecutar save_model_ucmerced.py primero")

# Modelos de datos para la API
class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    predicted_class: str = Field(..., description="Clase predicha")
    class_id: int = Field(..., description="ID de la clase (0-20)")
    confidence: float = Field(..., description="Confianza de la predicción (0-1)")
    top_3_predictions: List[Dict[str, float]] = Field(..., description="Top 3 predicciones más probables")

class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str
    model_loaded: bool
    model_path: str
    num_classes: int

class ModelInfoResponse(BaseModel):
    """Información del modelo"""
    dataset: str
    architecture: str
    input_shape: List[int]
    output_classes: int
    class_names: Dict[int, str]
    total_parameters: int

# ====================== ENDPOINTS ======================

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información básica de la API"""
    return {
        "message": "UC Merced Land Use Classification API",
        "dataset": "UC Merced Land Use Dataset (21 classes)",
        "documentation": "/docs",
        "health": "/health",
        "version": "2.0.0"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verificar el estado de la API y el modelo"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "num_classes": len(CLASS_NAMES)
    }

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Obtener información sobre el modelo cargado"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "dataset": "UC Merced Land Use Dataset",
        "architecture": "Convolutional Neural Network (CNN) - 4 Conv Blocks",
        "input_shape": [IMG_HEIGHT, IMG_WIDTH, 3],
        "output_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "total_parameters": model.count_params()
    }

@app.get("/classes", tags=["Model"])
async def get_classes():
    """Obtener la lista de clases que puede predecir el modelo"""
    return {
        "classes": CLASS_NAMES,
        "total_classes": len(CLASS_NAMES),
        "description": "21 land use categories from UC Merced dataset"
    }

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocesa la imagen para el modelo
    - Convierte a RGB (si es necesario)
    - Redimensiona a 128x128
    - Normaliza valores entre 0 y 1
    """
    try:
        # Leer imagen
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir a RGB si no lo es
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Convertir a array numpy
        image_array = np.array(image)
        
        # Normalizar (0-1)
        image_array = image_array.astype('float32') / 255.0
        
        # Reshape para el modelo (1, height, width, channels)
        image_array = image_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar imagen: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_image(file: UploadFile = File(..., description="Imagen de uso de tierra (será redimensionada a 128x128)")):
    """
    Predecir la categoría de uso de tierra de una imagen
    
    **Clases disponibles:**
    - agricultural, airplane, baseballdiamond, beach, buildings
    - chaparral, denseresidential, forest, freeway, golfcourse
    - harbor, intersection, mediumresidential, mobilehomepark
    - overpass, parkinglot, river, runway, sparseresidential
    - storagetanks, tenniscourt
    
    **Parámetros:**
    - **file**: Imagen en formato PNG, JPG, JPEG (preferiblemente 256x256 pero funciona con cualquier tamaño)
    
    **Retorna:**
    - Clase predicha
    - Confianza de la predicción
    - Top 3 predicciones más probables
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Validar tipo de archivo
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Formato de imagen no soportado. Use PNG o JPEG"
        )
    
    # Leer y preprocesar imagen
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    
    # Realizar predicción
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_id = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class_id])
    
    # Obtener top 3 predicciones
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {
            "class": CLASS_NAMES.get(int(idx), f"unknown_{idx}"),
            "confidence": float(predictions[0][idx])
        }
        for idx in top_3_indices
    ]
    
    return {
        "predicted_class": CLASS_NAMES.get(predicted_class_id, f"unknown_{predicted_class_id}"),
        "class_id": predicted_class_id,
        "confidence": confidence,
        "top_3_predictions": top_3_predictions
    }

@app.post("/batch/predict", tags=["Prediction"])
async def batch_predict(files: List[UploadFile] = File(..., description="Lista de imágenes para predicción en batch")):
    """
    Predecir múltiples imágenes en un solo request
    
    **Parámetros:**
    - **files**: Lista de imágenes (máximo 50)
    
    **Retorna:**
    - Lista de predicciones con top 3 para cada imagen
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Máximo 50 imágenes por batch")
    
    results = []
    
    for idx, file in enumerate(files):
        try:
            # Leer y preprocesar
            image_bytes = await file.read()
            processed_image = preprocess_image(image_bytes)
            
            # Predicción
            predictions = model.predict(processed_image, verbose=0)
            predicted_class_id = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_id])
            
            # Top 3
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3 = [
                {
                    "class": CLASS_NAMES.get(int(i), f"unknown_{i}"),
                    "confidence": float(predictions[0][i])
                }
                for i in top_3_indices
            ]
            
            results.append({
                "image_index": idx,
                "filename": file.filename,
                "predicted_class": CLASS_NAMES.get(predicted_class_id, f"unknown_{predicted_class_id}"),
                "class_id": predicted_class_id,
                "confidence": confidence,
                "top_3_predictions": top_3
            })
        except Exception as e:
            results.append({
                "image_index": idx,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"predictions": results, "total_images": len(files)}

@app.get("/examples", tags=["Model"])
async def get_example_classes():
    """
    Obtener ejemplos de cada clase con descripción
    """
    examples = {
        "agricultural": "Áreas agrícolas con cultivos y campos",
        "airplane": "Aviones en aeropuertos o estacionados",
        "baseballdiamond": "Campos de béisbol con diamante característico",
        "beach": "Playas y áreas costeras",
        "buildings": "Edificios y estructuras urbanas",
        "chaparral": "Vegetación de chaparral y arbustos",
        "denseresidential": "Áreas residenciales densamente pobladas",
        "forest": "Áreas forestales y bosques",
        "freeway": "Autopistas y carreteras principales",
        "golfcourse": "Campos de golf",
        "harbor": "Puertos y áreas portuarias",
        "intersection": "Intersecciones de carreteras",
        "mediumresidential": "Áreas residenciales de densidad media",
        "mobilehomepark": "Parques de casas móviles",
        "overpass": "Pasos elevados y puentes",
        "parkinglot": "Estacionamientos",
        "river": "Ríos y cursos de agua",
        "runway": "Pistas de aterrizaje",
        "sparseresidential": "Áreas residenciales poco densas",
        "storagetanks": "Tanques de almacenamiento",
        "tenniscourt": "Canchas de tenis"
    }
    
    return {"examples": examples, "total": len(examples)}

# ====================== MAIN ======================

if __name__ == "__main__":
    print("🚀 Iniciando UC Merced Land Use Classification API")
    print("📚 Documentación disponible en: http://localhost:8000/docs")
    print("📊 ReDoc disponible en: http://localhost:8000/redoc")
    
    uvicorn.run(
        "api_ucmerced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
