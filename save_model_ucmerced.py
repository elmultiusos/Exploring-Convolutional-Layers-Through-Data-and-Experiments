"""
Script para entrenar y guardar el modelo CNN para UC Merced Land Use Dataset
Dataset: 21 clases de uso de tierra, imágenes RGB 256x256
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import json

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuración
IMG_HEIGHT = 128  # Reducimos de 256 a 128 para entrenar más rápido
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20

def load_label_map(label_map_path='label_map.json'):
    """Carga el mapeo de etiquetas desde JSON"""
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    return label_map

def build_cnn_model(num_classes=21, kernel_size=(3, 3), img_height=128, img_width=128):
    """
    Construye el modelo CNN adaptado para UC Merced Land Use Dataset
    
    Arquitectura diseñada para imágenes RGB 128x128 con 21 clases
    """
    model = models.Sequential([
        # Primera capa de entrada - procesa imágenes RGB
        layers.Input(shape=(img_height, img_width, 3)),
        
        # Bloque Convolucional 1
        layers.Conv2D(32, kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque Convolucional 2
        layers.Conv2D(64, kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque Convolucional 3
        layers.Conv2D(128, kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque Convolucional 4 (más profundo para imágenes complejas)
        layers.Conv2D(256, kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten y capas densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model

def create_data_generators(data_dir='images_train_test_val', 
                          img_height=128, 
                          img_width=128,
                          batch_size=32):
    """
    Crea generadores de datos con data augmentation
    """
    # Data augmentation para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Solo rescaling para validación y test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generadores
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def train_and_save_model(data_dir='images_train_test_val',
                         epochs=20,
                         kernel_size=(3, 3),
                         model_path='models/ucmerced_cnn.h5',
                         img_height=128,
                         img_width=128,
                         batch_size=32):
    """
    Entrena el modelo y lo guarda en el path especificado
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print("="*60)
    print("🌍 UC MERCED LAND USE CLASSIFICATION - CNN Training")
    print("="*60)
    print(f"\nConfiguración:")
    print(f"  • Imagen: {img_height}x{img_width} RGB")
    print(f"  • Kernel size: {kernel_size}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Clases: 21")
    print()
    
    # Verificar si existe el directorio de datos
    if not os.path.exists(data_dir):
        print(f"❌ Error: No se encuentra el directorio '{data_dir}'")
        print(f"   Asegúrate de tener la estructura correcta de carpetas")
        return None, None
    
    # Crear generadores de datos
    print("📂 Cargando datos...")
    try:
        train_gen, val_gen, test_gen = create_data_generators(
            data_dir, img_height, img_width, batch_size
        )
        
        print(f"✓ Train samples: {train_gen.samples}")
        print(f"✓ Validation samples: {val_gen.samples}")
        print(f"✓ Test samples: {test_gen.samples}")
        print()
        
        # Mostrar mapeo de clases
        class_indices = train_gen.class_indices
        print("📋 Mapeo de clases:")
        for class_name, idx in sorted(class_indices.items(), key=lambda x: x[1]):
            print(f"  {idx:2d}: {class_name}")
        print()
        
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        print(f"   Verifica que el directorio '{data_dir}' tenga las carpetas train/validation/test")
        return None, None
    
    # Build model
    print(f"🏗️  Construyendo modelo CNN con kernel size {kernel_size}...")
    model = build_cnn_model(
        num_classes=21, 
        kernel_size=kernel_size,
        img_height=img_height,
        img_width=img_width
    )
    model.summary()
    print()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print(f"🚀 Entrenando modelo por {epochs} epochs...")
    print("="*60)
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    
    # Evaluate on test set
    print("\n📊 Evaluando en conjunto de prueba...")
    test_results = model.evaluate(test_gen, verbose=0)
    
    print(f"\n✓ Resultados finales:")
    print(f"  Test Loss: {test_results[0]:.4f}")
    print(f"  Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    print(f"  Test Top-3 Accuracy: {test_results[2]:.4f} ({test_results[2]*100:.2f}%)")
    
    # Save final model
    print(f"\n💾 Guardando modelo en {model_path}...")
    model.save(model_path)
    
    # Guardar también el mapeo de clases
    class_map_path = model_path.replace('.h5', '_classes.json')
    with open(class_map_path, 'w') as f:
        json.dump(class_indices, f, indent=2)
    print(f"💾 Mapeo de clases guardado en {class_map_path}")
    
    print("\n✅ ¡Entrenamiento completado exitosamente!")
    print("="*60)
    
    return model, history

if __name__ == "__main__":
    import sys
    
    # Verificar argumentos
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'images_train_test_val'
    
    # Entrenar y guardar el modelo
    model, history = train_and_save_model(
        data_dir=data_dir,
        epochs=20,
        kernel_size=(3, 3),
        model_path='models/ucmerced_cnn.h5',
        img_height=128,
        img_width=128,
        batch_size=32
    )
    
    if model:
        print("\n🎉 ¡Listo! Ahora puedes usar el modelo con la API.")
        print("   Ejecuta: python api.py")
