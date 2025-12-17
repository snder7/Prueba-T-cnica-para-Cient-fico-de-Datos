"""
Script para entrenar el modelo de predicción de rendimiento estudiantil.
Este script entrena el modelo SIN escalado para facilitar la integración con la API.
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent))

from data_processor import StudentDataProcessor
from model_trainer import StudentPerformancePredictor

def main():
    print("\n" + "="*80)
    print(" " * 20 + "ENTRENAMIENTO DE MODELO")
    print("="*80 + "\n")

    # PASO 1: Procesar datos
    print("PASO 1: Procesando datos...")
    processor = StudentDataProcessor()

    # IMPORTANTE: scale_features=False para evitar problemas con la API
    X_train, X_test, y_train, y_test, df_processed = processor.process_pipeline(
        source='url',
        engineer_features=True,
        scale_features=False,  # ← SIN ESCALADO para compatibilidad con API
        test_size=0.2,
        random_state=42
    )

    print(f"\n✓ Datos procesados correctamente")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {processor.get_feature_names()}")

    # PASO 2: Entrenar modelos
    print("\n" + "="*80)
    print("PASO 2: Entrenando modelos...")
    print("="*80 + "\n")

    predictor = StudentPerformancePredictor()
    results_df = predictor.train_all_models(X_train, y_train, X_test, y_test)

    # PASO 3: Mostrar resultados
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)
    print(results_df[['model_name', 'test_r2', 'test_mae', 'test_rmse', 'cv_r2_mean']].to_string(index=False))

    # PASO 4: Feature importance
    print("\n" + "="*80)
    print("IMPORTANCIA DE FEATURES")
    print("="*80)
    predictor.get_feature_importance(X_train, feature_names=processor.get_feature_names())

    # PASO 5: Guardar modelo
    print("\n" + "="*80)
    print("GUARDANDO MODELO...")
    print("="*80)

    # Cambiar ruta para guardar en la carpeta correcta
    model_path = Path(__file__).parent.parent / "models" / "trained_model.pkl"
    predictor.save_model(file_path=str(model_path), save_metadata=True)

    print("\n" + "="*80)
    print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*80)
    print(f"\nMejor modelo: {predictor.best_model_name}")
    print(f"Test R²: {results_df.iloc[0]['test_r2']:.4f}")
    print(f"Test MAE: {results_df.iloc[0]['test_mae']:.4f}")
    print(f"\nModelo guardado en: {model_path}")
    print("\nPuedes iniciar la API con: python api.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
