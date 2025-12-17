"""
====================================================================
MODEL TRAINER - STUDENT PERFORMANCE PREDICTION
====================================================================
Módulo para entrenamiento, evaluación y comparación de modelos
de Machine Learning para predicción de rendimiento estudiantil.

Modelos incluidos:
    - Linear Regression (baseline)
    - Random Forest Regressor
    - Gradient Boosting (XGBoost)
    - LightGBM
    - Support Vector Regression

Autor: Científico de Datos
Fecha: 2025
====================================================================
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path

# Machine Learning
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


class StudentPerformancePredictor:
    """
    Clase para entrenar y evaluar modelos de predicción de rendimiento estudiantil.

    Attributes:
        models (dict): Diccionario de modelos a entrenar
        best_model: Mejor modelo entrenado
        best_model_name (str): Nombre del mejor modelo
        model_scores (dict): Scores de todos los modelos
    """

    def __init__(self):
        """Inicializa el predictor con modelos predefinidos."""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_scores = {}
        self.feature_importance = None
        self._initialize_models()


    def _initialize_models(self):
        """Inicializa los modelos de ML a entrenar."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=-1
            )
        }
        print(f"✅ {len(self.models)} modelos inicializados")


    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """
        Evalúa un modelo en los conjuntos de entrenamiento y prueba.

        Args:
            model: Modelo de sklearn
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de prueba
            y_test: Target de prueba
            model_name (str): Nombre del modelo

        Returns:
            dict: Métricas de evaluación
        """
        # Predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Métricas en entrenamiento
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        # Métricas en prueba
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='r2', n_jobs=-1)

        metrics = {
            'model_name': model_name,
            'train_mae': round(train_mae, 4),
            'train_rmse': round(train_rmse, 4),
            'train_r2': round(train_r2, 4),
            'test_mae': round(test_mae, 4),
            'test_rmse': round(test_rmse, 4),
            'test_r2': round(test_r2, 4),
            'cv_r2_mean': round(cv_scores.mean(), 4),
            'cv_r2_std': round(cv_scores.std(), 4),
            'overfit_score': round(train_r2 - test_r2, 4)
        }

        return metrics


    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Entrena y evalúa todos los modelos.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_test: Features de prueba
            y_test: Target de prueba

        Returns:
            pd.DataFrame: Tabla comparativa de todos los modelos
        """
        print("\n" + "="*80)
        print(" " * 25 + "ENTRENAMIENTO DE MODELOS")
        print("="*80 + "\n")

        results = []

        for name, model in self.models.items():
            print(f"Entrenando {name}...")

            # Entrenar modelo
            model.fit(X_train, y_train)

            # Evaluar modelo
            metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test, name)
            results.append(metrics)

            print(f"   ✓ Test R²: {metrics['test_r2']:.4f} | Test MAE: {metrics['test_mae']:.4f} | Test RMSE: {metrics['test_rmse']:.4f}")

        # Crear DataFrame con resultados
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('test_r2', ascending=False).reset_index(drop=True)

        # Guardar scores
        self.model_scores = results_df.to_dict('records')

        # Identificar mejor modelo
        best_model_row = results_df.iloc[0]
        self.best_model_name = best_model_row['model_name']
        self.best_model = self.models[self.best_model_name]

        print("\n" + "="*80)
        print(f"🏆 MEJOR MODELO: {self.best_model_name}")
        print(f"   Test R²: {best_model_row['test_r2']:.4f}")
        print(f"   Test MAE: {best_model_row['test_mae']:.4f}")
        print(f"   Test RMSE: {best_model_row['test_rmse']:.4f}")
        print("="*80)

        return results_df


    def hyperparameter_tuning(self, X_train, y_train, model_type='random_forest'):
        """
        Realiza búsqueda de hiperparámetros óptimos.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            model_type (str): Tipo de modelo ('random_forest', 'xgboost', 'lightgbm')

        Returns:
            Best estimator y mejores parámetros
        """
        print(f"\n🔍 Optimizando hiperparámetros para {model_type}...")

        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)

        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

        elif model_type == 'lightgbm':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'num_leaves': [15, 31, 63]
            }
            model = lgb.LGBMRegressor(random_state=42, verbose=-1)

        else:
            raise ValueError("model_type debe ser 'random_forest', 'xgboost' o 'lightgbm'")

        # Grid Search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"✅ Mejor score (CV R²): {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_


    def get_feature_importance(self, X_train, feature_names=None):
        """
        Obtiene la importancia de las features del mejor modelo.

        Args:
            X_train: Features de entrenamiento
            feature_names (list): Nombres de las features

        Returns:
            pd.DataFrame: Importancia de features ordenada
        """
        if self.best_model is None:
            raise ValueError("Primero debes entrenar los modelos con train_all_models()")

        # Verificar si el modelo tiene feature_importances_
        if hasattr(self.best_model, 'feature_importances_'):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False).reset_index(drop=True)

            self.feature_importance = importance_df

            print("\n" + "="*70)
            print("IMPORTANCIA DE FEATURES (Top 10)")
            print("="*70)
            print(importance_df.head(10).to_string(index=False))

            return importance_df

        else:
            print(f"⚠️  El modelo {self.best_model_name} no tiene feature_importances_")
            return None


    def save_model(self, file_path='../models/trained_model.pkl', save_metadata=True):
        """
        Guarda el mejor modelo entrenado.

        Args:
            file_path (str): Ruta donde guardar el modelo
            save_metadata (bool): Si True, guarda también metadatos

        Returns:
            str: Ruta del modelo guardado
        """
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar")

        # Crear directorio si no existe
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Guardar modelo
        joblib.dump(self.best_model, file_path)
        print(f"\n✅ Modelo guardado en: {file_path}")

        # Guardar metadatos
        if save_metadata:
            metadata = {
                'model_name': self.best_model_name,
                'model_scores': self.model_scores,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None
            }

            metadata_path = file_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            print(f"✅ Metadatos guardados en: {metadata_path}")

        return file_path


    def load_model(self, file_path='../models/trained_model.pkl'):
        """
        Carga un modelo previamente entrenado.

        Args:
            file_path (str): Ruta del modelo

        Returns:
            Model: Modelo cargado
        """
        try:
            self.best_model = joblib.load(file_path)
            print(f"✅ Modelo cargado desde: {file_path}")

            # Intentar cargar metadatos
            metadata_path = file_path.replace('.pkl', '_metadata.json')
            if Path(metadata_path).exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.best_model_name = metadata.get('model_name', 'Unknown')
                print(f"   Modelo: {self.best_model_name}")
                print(f"   Fecha de entrenamiento: {metadata.get('training_date', 'Unknown')}")

            return self.best_model

        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {file_path}")
            raise


    def predict(self, X):
        """
        Realiza predicciones con el mejor modelo.

        Args:
            X: Features para predicción

        Returns:
            np.array: Predicciones
        """
        if self.best_model is None:
            raise ValueError("Primero debes entrenar o cargar un modelo")

        predictions = self.best_model.predict(X)
        return predictions


# ====================================================================
# FUNCIÓN AUXILIAR PARA USO RÁPIDO
# ====================================================================

def quick_train(X_train, y_train, X_test, y_test, save_model=True):
    """
    Función de acceso rápido para entrenar modelos.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_test: Features de prueba
        y_test: Target de prueba
        save_model (bool): Si True, guarda el mejor modelo

    Returns:
        StudentPerformancePredictor: Predictor entrenado
    """
    predictor = StudentPerformancePredictor()
    results_df = predictor.train_all_models(X_train, y_train, X_test, y_test)

    if save_model:
        predictor.save_model()

    return predictor, results_df


# ====================================================================
# EJEMPLO DE USO
# ====================================================================

if __name__ == "__main__":
    from data_processor import quick_process

    print("\n" + "="*80)
    print(" " * 25 + "EJECUTANDO MODEL TRAINER")
    print("="*80 + "\n")

    # 1. Procesar datos
    print("PASO 1: Procesando datos...")
    X_train, X_test, y_train, y_test, processor = quick_process(
        source='url',
        engineer_features=True
    )

    # 2. Entrenar modelos
    print("\nPASO 2: Entrenando modelos...")
    predictor = StudentPerformancePredictor()
    results_df = predictor.train_all_models(X_train, y_train, X_test, y_test)

    # 3. Mostrar resultados
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)
    print(results_df.to_string(index=False))

    # 4. Feature importance
    predictor.get_feature_importance(X_train, feature_names=processor.get_feature_names())

    # 5. Guardar modelo
    predictor.save_model()

    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
