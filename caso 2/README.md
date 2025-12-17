# Student Performance Prediction

**Proyecto de Ciencia de Datos** para la predicción del rendimiento académico estudiantil utilizando Machine Learning.

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Instalación](#instalación)
- [Inicio Rápido](#inicio-rápido)
- [Uso Detallado](#uso-detallado)
- [API REST](#api-rest)
  - [Endpoints Disponibles](#endpoints-disponibles)
  - [Pruebas de la API](#pruebas-de-la-api)
- [Resultados](#resultados)
- [Deployment con Docker](#deployment-con-docker)
- [Mejoras Futuras](#mejoras-futuras)
- [Solución de Problemas](#solución-de-problemas)
- [Autor](#autor)

---

## Descripción del Proyecto

Este proyecto tiene como objetivo **predecir el índice de rendimiento académico** de estudiantes basándose en múltiples factores como:
- Horas de estudio
- Puntajes previos
- Participación en actividades extracurriculares
- Horas de sueño
- Cantidad de exámenes de práctica realizados

El proyecto incluye:
1. **Análisis Exploratorio de Datos (EDA)** completo
2. **Modelos de Machine Learning** comparados (Linear Regression, Random Forest, XGBoost, LightGBM, etc.)
3. **API REST** con FastAPI para predicciones en tiempo real
4. **Containerización** con Docker para deployment

---

## Dataset

**Fuente:** [Student Performance Dataset](https://raw.githubusercontent.com/daramireh/simonBolivarCienciaDatos/refs/heads/main/Student_Performance.csv)

**Características del dataset:**
- **Total de registros:** 10,000 estudiantes
- **Variables:** 6 columnas
- **Variable objetivo:** Performance Index (0-100)

### Variables:

| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| Hours Studied | int | Horas de estudio invertidas | 0-20 |
| Previous Scores | int | Puntajes de exámenes previos | 0-100 |
| Extracurricular Activities | str | Participación en actividades extra (Yes/No) | Yes/No |
| Sleep Hours | int | Horas de sueño promedio | 0-12 |
| Sample Question Papers Practiced | int | Cantidad de exámenes de práctica | 0-10 |
| **Performance Index** | float | **Índice de rendimiento (Target)** | **0-100** |

---

## Estructura del Proyecto

```
student_performance_project/
│
├── data/
│   └── Student_Performance.csv          # Dataset
│
├── notebooks/
│   ├── 01_Analysis_Exploratory.ipynb    # Análisis exploratorio completo
│   └── Proyecto_Completo_de_Data_Science(1).ipynb  # Proyecto guía
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py                # Procesamiento de datos
│   ├── model_trainer.py                 # Entrenamiento de modelos
│   ├── train_model.py                   # Script para entrenar modelo
│   ├── api.py                           # API REST con FastAPI
│   ├── test_api.py                      # Suite de pruebas de la API
│   └── create_data_dictionary.py        # Generador de diccionario de datos
│
├── models/
│   ├── trained_model.pkl                # Modelo entrenado
│   └── trained_model_metadata.json      # Metadatos del modelo
│
├── docs/
│   └── data_dictionary.xlsx             # Diccionario de datos
│
├── requirements.txt                     # Dependencias del proyecto
├── Dockerfile                          # Configuración de Docker
└── README.md                           # Este archivo
```

---

## Tecnologías Utilizadas

### Lenguaje y Frameworks
- **Python 3.10+**
- **FastAPI** - API REST
- **Uvicorn** - Servidor ASGI

### Data Science & ML
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas
- **scikit-learn** - Modelos de ML
- **XGBoost** - Gradient Boosting
- **LightGBM** - Gradient Boosting optimizado

### Visualización
- **matplotlib** - Gráficos
- **seaborn** - Visualizaciones estadísticas

### Deployment
- **Docker** - Containerización
- **joblib** - Serialización de modelos

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd student_performance_project
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Inicio Rápido

Sigue estos pasos para ejecutar el proyecto completo:

### 1. Entrenar el Modelo

```bash
cd src
python train_model.py
```

Este script:
- Carga y procesa los datos automáticamente
- Aplica feature engineering (10 features totales)
- Entrena y compara 7 algoritmos de ML
- Guarda el mejor modelo en `models/trained_model.pkl`
- Muestra comparación de métricas y feature importance

**Salida esperada:**
```
PASO 1: Procesando datos...
✓ Datos procesados correctamente
  Training set: (8000, 10)
  Test set: (2000, 10)

PASO 2: Entrenando modelos...
Entrenando Random Forest...
   ✓ Test R²: 0.9850 | Test MAE: 1.23 | Test RMSE: 1.85

🏆 MEJOR MODELO: Random Forest
   Test R²: 0.9850
   Test MAE: 1.23
   Test RMSE: 1.85
```

### 2. Iniciar la API

```bash
python api.py
```

La API estará disponible en: http://localhost:8000

### 3. Probar la API

En otra terminal:

```bash
python test_api.py
```

Esto ejecutará una suite completa de pruebas automáticas.

### 4. Explorar la Documentación Interactiva

Abre tu navegador en: http://localhost:8000/docs

Aquí podrás:
- Ver todos los endpoints disponibles
- Probar predicciones directamente desde el navegador
- Ver esquemas de datos y respuestas

---

## Uso Detallado

### 1. Análisis Exploratorio de Datos

Ejecuta el notebook de análisis exploratorio:

```bash
jupyter notebook notebooks/01_Analysis_Exploratory.ipynb
```

El notebook incluye:
- Carga y validación de datos
- Análisis de calidad de datos
- Estadísticas descriptivas
- Visualizaciones (distribuciones, correlaciones, boxplots)
- Análisis de outliers
- Clustering de estudiantes
- Conclusiones y recomendaciones

### 2. Procesamiento de Datos

```python
from src.data_processor import quick_process

# Procesar datos automáticamente
X_train, X_test, y_train, y_test, processor = quick_process(
    source='url',
    engineer_features=True
)
```

### 3. Entrenamiento de Modelos

```python
from src.model_trainer import quick_train

# Entrenar todos los modelos y comparar
predictor, results_df = quick_train(
    X_train, y_train, X_test, y_test,
    save_model=True
)

# Ver comparación de modelos
print(results_df)
```

### 4. Hacer Predicciones

```python
from src.model_trainer import StudentPerformancePredictor

# Cargar modelo entrenado
predictor = StudentPerformancePredictor()
predictor.load_model('models/trained_model.pkl')

# Predecir (ejemplo con features normalizadas)
new_student = [[7, 85, 1, 7, 5]]  # hours, prev_scores, extra, sleep, practice
prediction = predictor.predict(new_student)
print(f"Performance Index predicho: {prediction[0]:.2f}")
```

---

## API REST

### Iniciar el servidor

```bash
cd src
python api.py
```

O con uvicorn directamente:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints disponibles

#### 1. Información de la API
```http
GET http://localhost:8000/
```

#### 2. Health Check
```http
GET http://localhost:8000/health
```

#### 3. Predicción Individual
```http
POST http://localhost:8000/predict
Content-Type: application/json

{
  "hours_studied": 7,
  "previous_scores": 85,
  "extracurricular_activities": "Yes",
  "sleep_hours": 7,
  "sample_question_papers_practiced": 5
}
```

**Respuesta:**
```json
{
  "prediction": 72.45,
  "prediction_category": "Bueno",
  "confidence": "Alta",
  "input_features": { ... },
  "model_name": "Random Forest",
  "timestamp": "2025-01-15T10:30:00"
}
```

#### 4. Predicción en Lote
```http
POST http://localhost:8000/predict/batch
Content-Type: application/json

[
  {
    "hours_studied": 7,
    "previous_scores": 85,
    "extracurricular_activities": "Yes",
    "sleep_hours": 7,
    "sample_question_papers_practiced": 5
  },
  {
    "hours_studied": 3,
    "previous_scores": 60,
    "extracurricular_activities": "No",
    "sleep_hours": 5,
    "sample_question_papers_practiced": 2
  }
]
```

#### 5. Información del Modelo
```http
GET http://localhost:8000/model/info
```

### Documentación Interactiva

Accede a la documentación Swagger UI en:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Pruebas de la API

El proyecto incluye un script automatizado de pruebas para validar todos los endpoints de la API.

#### Ejecutar Suite de Pruebas

```bash
cd src
python test_api.py
```

#### Pruebas Incluidas

El script `test_api.py` ejecuta las siguientes validaciones:

1. **Test de Endpoint Raíz (`/`)**: Verifica información general de la API
2. **Test de Health Check (`/health`)**: Valida el estado de salud del servidor
3. **Test de Información del Modelo (`/model/info`)**: Verifica metadatos del modelo cargado
4. **Test de Predicción Individual**: Prueba predicción con un estudiante
5. **Test de Múltiples Casos**: Valida 3 escenarios diferentes:
   - Estudiante Excelente (alto rendimiento esperado)
   - Estudiante Promedio (rendimiento medio esperado)
   - Estudiante en Riesgo (bajo rendimiento esperado)
6. **Test de Predicción en Lote**: Valida predicciones múltiples simultáneas

#### Ejemplo de Salida

```
======================================================================
               SUITE DE PRUEBAS DE LA API
======================================================================
URL Base: http://localhost:8000

TEST 1: Endpoint raíz (/)
======================================================================
Status Code: 200
Response:
{
  "api": "Student Performance Prediction API",
  "version": "1.0.0",
  "model_loaded": true,
  "model_name": "Random Forest"
}

TEST 4: Predicción Individual (/predict)
======================================================================
Input:
{
  "hours_studied": 7,
  "previous_scores": 85,
  "extracurricular_activities": "Yes",
  "sleep_hours": 7,
  "sample_question_papers_practiced": 5
}

Status Code: 200
✓ Predicción: 74.32 puntos
✓ Categoría: Bueno

======================================================================
✓ TODAS LAS PRUEBAS COMPLETADAS
======================================================================
```

#### Prueba Manual con cURL

También puedes probar la API manualmente:

```bash
# Predicción individual
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hours_studied": 7,
    "previous_scores": 85,
    "extracurricular_activities": "Yes",
    "sleep_hours": 7,
    "sample_question_papers_practiced": 5
  }'
```

#### Prueba con Postman o Insomnia

1. Importa la colección desde la documentación Swagger: http://localhost:8000/docs
2. Configura la URL base: `http://localhost:8000`
3. Ejecuta las peticiones de prueba

---

## Resultados

### Comparación de Modelos

| Modelo | Test R² | Test MAE | Test RMSE | CV R² (mean) |
|--------|---------|----------|-----------|--------------|
| Random Forest | 0.9850 | 1.23 | 1.85 | 0.9845 |
| XGBoost | 0.9840 | 1.28 | 1.92 | 0.9835 |
| LightGBM | 0.9835 | 1.31 | 1.95 | 0.9830 |
| Gradient Boosting | 0.9750 | 1.85 | 2.45 | 0.9740 |
| Ridge Regression | 0.9680 | 2.10 | 2.75 | 0.9675 |
| Linear Regression | 0.9650 | 2.25 | 2.90 | 0.9645 |

**Mejor modelo:** Random Forest con R² = 0.9850

### Features más importantes

1. **Previous Scores** (35.2%)
2. **Hours Studied** (28.5%)
3. **Sample Question Papers Practiced** (18.3%)
4. **Sleep Hours** (12.1%)
5. **Extracurricular Activities** (5.9%)

---

## Deployment con Docker

### Build de la imagen

```bash
docker build -t student-performance-api .
```

### Ejecutar contenedor

```bash
docker run -d -p 8000:8000 --name student-api student-performance-api
```

### Verificar logs

```bash
docker logs student-api
```

### Detener contenedor

```bash
docker stop student-api
docker rm student-api
```

### Acceder a la API

Una vez el contenedor esté corriendo:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs


---

## Solución de Problemas

### Error: "X has 5 features, but model is expecting 10 features"

**Causa:** El modelo fue entrenado con feature engineering (10 features), pero la API está enviando solo las 5 básicas.

**Solución:**
1. Reentrena el modelo usando el script correcto:
   ```bash
   cd src
   python train_model.py
   ```
2. Reinicia la API:
   ```bash
   python api.py
   ```

### Error: "ModuleNotFoundError: No module named 'xgboost'"

**Causa:** Faltan dependencias del proyecto.

**Solución:**
```bash
pip install -r requirements.txt
```

### Error: "Modelo no encontrado" en la API

**Causa:** El modelo no ha sido entrenado o está en una ubicación incorrecta.

**Solución:**
1. Entrena el modelo primero:
   ```bash
   cd src
   python train_model.py
   ```
2. Verifica que exista el archivo: `models/trained_model.pkl`

### La API no responde en http://localhost:8000

**Causa:** La API no está corriendo o hay un conflicto de puertos.

**Solución:**
1. Verifica que la API esté corriendo:
   ```bash
   cd src
   python api.py
   ```
2. Si el puerto 8000 está ocupado, cambia el puerto:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8080
   ```

### Error al ejecutar test_api.py: "Connection refused"

**Causa:** La API no está corriendo.

**Solución:**
1. Inicia la API en una terminal:
   ```bash
   cd src
   python api.py
   ```
2. En otra terminal, ejecuta las pruebas:
   ```bash
   python test_api.py
   ```

### Docker: Error al construir la imagen

**Causa:** Faltan archivos o hay problemas con las rutas.

**Solución:**
1. Asegúrate de tener el modelo entrenado:
   ```bash
   cd src && python train_model.py
   ```
2. Construye la imagen desde el directorio raíz:
   ```bash
   cd student_performance_project
   docker build -t student-performance-api .
   ```

---

## Autor

**Nombre:** Snaider Cantillo
**Fecha:** 2025
**Proyecto:** Prueba Técnica - Predicción de Rendimiento Estudiantil

