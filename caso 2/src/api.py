"""
====================================================================
API REST - STUDENT PERFORMANCE PREDICTION
====================================================================
API REST construida con FastAPI para realizar predicciones de
rendimiento estudiantil usando modelo de Machine Learning entrenado.

Endpoints:
    - GET  /: Información de la API
    - GET  /health: Health check
    - POST /predict: Predicción individual
    - POST /predict/batch: Predicción en lote
    - GET  /model/info: Información del modelo

Autor: Científico de Datos
Fecha: 2025
====================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ====================================================================
# CONFIGURACIÓN DE LA API
# ====================================================================

app = FastAPI(
    title="Student Performance Prediction API",
    description="API para predicción de rendimiento académico estudiantil usando Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================================================================
# MODELOS DE DATOS (Pydantic)
# ====================================================================

class StudentFeatures(BaseModel):
    """
    Modelo de datos para features del estudiante.
    """
    hours_studied: int = Field(
        ...,
        ge=0,
        le=20,
        description="Horas de estudio (0-20)",
        example=7
    )
    previous_scores: int = Field(
        ...,
        ge=0,
        le=100,
        description="Puntajes previos (0-100)",
        example=85
    )
    extracurricular_activities: str = Field(
        ...,
        description="Participación en actividades extracurriculares (Yes/No)",
        example="Yes"
    )
    sleep_hours: int = Field(
        ...,
        ge=0,
        le=12,
        description="Horas de sueño promedio (0-12)",
        example=7
    )
    sample_question_papers_practiced: int = Field(
        ...,
        ge=0,
        le=10,
        description="Cantidad de exámenes de práctica (0-10)",
        example=5
    )

    @validator('extracurricular_activities')
    def validate_extracurricular(cls, v):
        """Valida que la actividad extracurricular sea Yes o No."""
        if v not in ['Yes', 'No']:
            raise ValueError('extracurricular_activities debe ser "Yes" o "No"')
        return v


class PredictionResponse(BaseModel):
    """
    Modelo de respuesta para predicciones individuales.
    """
    prediction: float = Field(..., description="Índice de rendimiento predicho (0-100)")
    prediction_category: str = Field(..., description="Categoría de rendimiento")
    confidence: str = Field(..., description="Nivel de confianza")
    input_features: dict = Field(..., description="Features utilizados para la predicción")
    model_name: str = Field(..., description="Nombre del modelo utilizado")
    timestamp: str = Field(..., description="Timestamp de la predicción")


class BatchPredictionResponse(BaseModel):
    """
    Modelo de respuesta para predicciones en lote.
    """
    predictions: List[float] = Field(..., description="Lista de predicciones")
    prediction_categories: List[str] = Field(..., description="Categorías de rendimiento")
    total_predictions: int = Field(..., description="Total de predicciones realizadas")
    model_name: str = Field(..., description="Nombre del modelo utilizado")
    timestamp: str = Field(..., description="Timestamp de la predicción")


class ModelInfo(BaseModel):
    """
    Modelo de información del modelo cargado.
    """
    model_name: str
    training_date: str
    model_path: str
    is_loaded: bool
    metadata: Optional[dict] = None


# ====================================================================
# CARGA DEL MODELO
# ====================================================================

from pathlib import Path
import joblib

class ModelLoader:
    """Clase para cargar y gestionar el modelo de ML."""

    def __init__(self, model_path=None):
        base_dir = Path(__file__).resolve().parent.parent

        self.model_path = (
            Path(model_path)
            if model_path
            else base_dir / "models" / "trained_model.pkl"
        )

        # ⚠️ atributos SIEMPRE definidos
        self.model = None
        self.metadata = None
        self.model_name = "Student Performance Model"

        self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            print(f"⚠️ Modelo no encontrado en: {self.model_path}")
            return

        self.model = joblib.load(self.model_path)
        print(f"✅ Modelo cargado desde: {self.model_path}")

        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones.

        Args:
            features (np.ndarray): Array de features

        Returns:
            np.ndarray: Predicciones

        Raises:
            HTTPException: Si el modelo no está cargado
        """
        if self.model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo no disponible. Por favor entrena y carga un modelo primero."
            )

        return self.model.predict(features)


# Inicializar cargador de modelo
model_loader = ModelLoader()


# ====================================================================
# FUNCIONES AUXILIARES
# ====================================================================

def preprocess_features(features: StudentFeatures) -> np.ndarray:
    """
    Preprocesa las features para el modelo, incluyendo feature engineering.
    IMPORTANTE: Debe coincidir exactamente con el preprocesamiento usado en el entrenamiento.

    Args:
        features (StudentFeatures): Features del estudiante

    Returns:
        np.ndarray: Array de features procesado con feature engineering
    """
    # Features básicas
    hours_studied = features.hours_studied
    previous_scores = features.previous_scores
    extracurricular = 1 if features.extracurricular_activities == 'Yes' else 0
    sleep_hours = features.sleep_hours
    sample_papers = features.sample_question_papers_practiced

    # ====================================================================
    # FEATURE ENGINEERING (mismo proceso que en data_processor.py)
    # ====================================================================

    # 1. Study_Practice_Interaction
    study_practice_interaction = hours_studied * sample_papers

    # 2. Study_Efficiency (evitar división por cero)
    study_efficiency = previous_scores / hours_studied if hours_studied > 0 else 0

    # 3. Low_Previous_Performance (umbral del percentil 25 ≈ 55)
    threshold_prev_score = 55.0  # Aproximado del percentil 25
    low_previous_performance = 1 if previous_scores < threshold_prev_score else 0

    # 4. Total_Study_Effort
    total_study_effort = hours_studied + sample_papers

    # 5. Sleep_Quality (Poor=0, Moderate=1, Good=2)
    if sleep_hours <= 5:
        sleep_quality = 0  # Poor
    elif sleep_hours <= 7:
        sleep_quality = 1  # Moderate
    else:
        sleep_quality = 2  # Good

    # ====================================================================
    # Crear array con TODAS las features (10 en total)
    # Orden: [5 básicas] + [5 de feature engineering]
    # ====================================================================
    features_array = np.array([[
        hours_studied,                  # 0
        previous_scores,                # 1
        extracurricular,                # 2
        sleep_hours,                    # 3
        sample_papers,                  # 4
        study_practice_interaction,     # 5
        study_efficiency,               # 6
        low_previous_performance,       # 7
        total_study_effort,             # 8
        sleep_quality                   # 9
    ]])

    return features_array


def categorize_performance(score: float) -> str:
    """
    Categoriza el rendimiento según el score.

    Args:
        score (float): Score de rendimiento (0-100)

    Returns:
        str: Categoría de rendimiento
    """
    if score >= 80:
        return "Excelente"
    elif score >= 60:
        return "Bueno"
    elif score >= 40:
        return "Regular"
    else:
        return "Bajo - Requiere intervención"


def get_confidence_level(score: float) -> str:
    """
    Determina el nivel de confianza basado en el score.

    Args:
        score (float): Score de rendimiento

    Returns:
        str: Nivel de confianza
    """
    # En un escenario real, esto se basaría en métricas del modelo
    if 0 <= score <= 100:
        return "Alta"
    else:
        return "Media"


# ====================================================================
# ENDPOINTS
# ====================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Endpoint raíz - Información de la API.

    Returns:
        dict: Información básica de la API
    """
    return {
        "api": "Student Performance Prediction API",
        "version": "1.0.0",
        "description": "API para predicción de rendimiento académico estudiantil",
        "model_loaded": model_loader.model is not None,
        "model_name": model_loader.model_name,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["General"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Estado de salud de la API
    """
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(features: StudentFeatures):
    """
    Realiza una predicción individual del rendimiento estudiantil.

    Args:
        features (StudentFeatures): Características del estudiante

    Returns:
        PredictionResponse: Predicción y metadatos

    Raises:
        HTTPException: Si el modelo no está cargado o hay error en la predicción
    """
    try:
        # Preprocesar features
        X = preprocess_features(features)

        # Realizar predicción
        prediction = model_loader.predict(X)[0]

        # Asegurar que la predicción esté en el rango [0, 100]
        prediction = float(np.clip(prediction, 0, 100))

        # Categorizar rendimiento
        category = categorize_performance(prediction)

        # Obtener nivel de confianza
        confidence = get_confidence_level(prediction)

        return PredictionResponse(
            prediction=round(prediction, 2),
            prediction_category=category,
            confidence=confidence,
            input_features=features.dict(),
            model_name=model_loader.model_name,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(features_list: List[StudentFeatures]):
    """
    Realiza predicciones en lote.

    Args:
        features_list (List[StudentFeatures]): Lista de características de estudiantes

    Returns:
        BatchPredictionResponse: Lista de predicciones y metadatos

    Raises:
        HTTPException: Si el modelo no está cargado o hay error en la predicción
    """
    try:
        if len(features_list) == 0:
            raise HTTPException(status_code=400, detail="La lista de features no puede estar vacía")

        # Preprocesar todas las features
        X_batch = np.vstack([preprocess_features(features) for features in features_list])

        # Realizar predicciones
        predictions = model_loader.predict(X_batch)

        # Asegurar que las predicciones estén en el rango [0, 100]
        predictions = np.clip(predictions, 0, 100)

        # Categorizar rendimientos
        categories = [categorize_performance(pred) for pred in predictions]

        return BatchPredictionResponse(
            predictions=[round(float(pred), 2) for pred in predictions],
            prediction_categories=categories,
            total_predictions=len(predictions),
            model_name=model_loader.model_name,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción batch: {str(e)}")


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Obtiene información del modelo cargado.

    Returns:
        ModelInfo: Información detallada del modelo
    """
    return ModelInfo(
        model_name=model_loader.model_name,
        training_date=model_loader.metadata.get('training_date', 'Unknown') if model_loader.metadata else 'Unknown',
        model_path=str(model_loader.model_path),  # Convertir Path a string
        is_loaded=model_loader.model is not None,
        metadata=model_loader.metadata
    )


# ====================================================================
# EJECUCIÓN DE LA API
# ====================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*70)
    print(" " * 20 + "INICIANDO API REST")
    print("="*70)
    print(f"Modelo cargado: {model_loader.model is not None}")
    print(f"Modelo: {model_loader.model_name}")
    print("\nAccede a la documentación en: http://localhost:8000/docs")
    print("="*70 + "\n")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
