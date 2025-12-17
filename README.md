# README — Prueba Técnica para Científico de Datos

Este repositorio/documento acompaña la **presentación en PDF** de la *Prueba Técnica para Científico de Datos* y describe de forma clara, estructurada y profesional el enfoque, la metodología y los entregables desarrollados para **los dos casos analíticos solicitados**.

El objetivo de este README es servir como **guía de lectura y sustentación técnica** durante la revisión del PDF y la entrevista técnica.

---

## 📌 Contenido de la Prueba

La prueba se compone de **dos casos independientes**, desarrollados en Python, que evalúan habilidades de:

- Análisis exploratorio de datos (EDA)
- Limpieza y transformación de datos
- Análisis estadístico
- Modelado predictivo (Machine Learning)
- Pensamiento analítico y de negocio
- Desarrollo de productos de datos (API)

---

## 📂 Caso 1 — Análisis de la Copa Mundial Femenina

### 🎯 Objetivo
Analizar la evolución histórica del rendimiento de las selecciones participantes en la **Copa Mundial Femenina de la FIFA (1991–2023)**, identificando patrones de desempeño, tendencias de goles y equipos dominantes.

### 📊 Datasets utilizados

- `world_cup_women.csv`  
  Información general de las ediciones del mundial (año, sede, equipos, goleadoras, asistencia).

- `matches_1991_2023.csv`  
  Información detallada de los partidos disputados (resultados, goles, tarjetas, asistencia).

Ambos datasets se consumen directamente desde URLs públicas (raw GitHub), garantizando reproducibilidad.

---

### 🧪 Actividades desarrolladas

1. **Análisis estructural de datos**  
   - Identificación de variables y tipos de datos
   - Detección de valores nulos y registros duplicados

2. **Validación cruzada entre tablas**  
   - Identificación de campos relacionales
   - Verificación de integridad referencial y datos faltantes

3. **Tabla de posiciones — Mundial 1991**  
   Se construyó una tabla de posiciones considerando:
   - 3 puntos por victoria
   - 1 punto por empate
   - Juego limpio:  
     - Tarjeta amarilla = -1 punto  
     - Tarjeta roja = -2 puntos

   **Estructura final:**
   
   Equipo | PJ | PG | PE | PP | GF | GC | DG | JL | Puntos

4. **Tabla de goleadoras — Mundial 2023**  
   Identificación de las máximas anotadoras y análisis de su impacto en el torneo.

5. **Tabla consolidada histórica**  
   Construcción de una única tabla agregada con métricas por año, sede y equipo:
   - Goles marcados y recibidos (totales y promedios)
   - Partidos ganados, perdidos y empatados
   - Promedio de asistencia por equipo

---

### 📈 Valor analítico

Este caso demuestra capacidad para:

- Integrar múltiples fuentes de datos
- Aplicar reglas de negocio complejas
- Generar indicadores comparables entre ediciones
- Analizar la evolución del fútbol femenino desde una perspectiva cuantitativa

---

## 📂 Caso 2 — Factores que Impactan el Desempeño en Matemáticas

### 🎯 Objetivo
Identificar los factores que influyen en el rendimiento académico en matemáticas y desarrollar un **producto de datos** que permita detectar tempranamente estudiantes con bajo desempeño.

### 📊 Dataset utilizado

- `Student_Performance.csv`

Contiene información sobre:
- Horas de estudio
- Calificaciones previas
- Actividades extracurriculares
- Horas de sueño
- Ejercicios prácticos realizados
- Índice de desempeño final (*Performance Index*)

---

### 🧪 Actividades desarrolladas

1. **Análisis de estructuras de datos**  
   - Visualizaciones descriptivas y correlacionales
   - Diccionario de datos con tipos de variables
   - Evaluación y aplicación de transformaciones necesarias

2. **Análisis exploratorio (EDA)**  
   - Estadística descriptiva completa
   - Identificación de patrones y relaciones entre variables

3. **Análisis estadístico inferencial**  
   - Evaluación de diferencias significativas en el Performance Index según asistencia a actividades extracurriculares
   - Aplicación de pruebas estadísticas (t-test / Mann-Whitney)

4. **Segmentación de estudiantes (Clustering)**  
   - Identificación de perfiles de estudiantes mediante algoritmos de agrupamiento

---

### 🤖 Modelado Predictivo

Se desarrollaron **dos tipos de modelos**:

- **Regresión**: predicción del índice de desempeño académico
- **Clasificación**: identificación de estudiantes con bajo rendimiento

Cada modelo fue comparado con **al menos dos alternativas adicionales**, utilizando métricas adecuadas:

- Regresión: RMSE, MAE, R²
- Clasificación: Recall, F1-score, AUC-ROC

El criterio de selección prioriza la **detección temprana de estudiantes en riesgo**.

---

## 🚀 Producto de Datos — API de Predicción

Como resultado del Caso 2, se desarrolló una **API REST en FastAPI** que permite:

- Consumir modelos entrenados
- Predecir rendimiento académico
- Identificar estudiantes con bajo desempeño
- Simular operaciones CRUD mediante diccionarios en memoria

### Características técnicas:

- Python + FastAPI
- Documentación automática (Swagger)
- Preparada para ejecución con Docker (o entorno virtual)

---

## 🧠 Enfoque Profesional

Esta prueba fue desarrollada siguiendo buenas prácticas de ciencia de datos:

- Reproducibilidad
- Separación entre análisis, modelado y despliegue
- Interpretabilidad de resultados
- Enfoque en impacto de negocio y toma de decisiones

Todo el contenido presentado puede ser **sustentado técnicamente** durante la entrevista.

---

---

**Autor:** Snaider Cantillo


