"""
====================================================================
DATA PROCESSOR - STUDENT PERFORMANCE PREDICTION
====================================================================
Módulo para procesamiento y transformación de datos del dataset
de rendimiento estudiantil.

Funciones principales:
    - load_data(): Carga el dataset
    - preprocess_data(): Preprocesamiento completo
    - engineer_features(): Creación de features adicionales
    - split_data(): División train/test

Autor: Científico de Datos
Fecha: 2025
====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class StudentDataProcessor:
    """
    Clase para procesamiento de datos de rendimiento estudiantil.

    Attributes:
        scaler (StandardScaler): Escalador para normalización de features
        label_encoder (LabelEncoder): Codificador para variables categóricas
        feature_names (list): Nombres de las features procesadas
    """

    def __init__(self):
        """Inicializa el procesador de datos."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []


    def load_data(self, source='url', file_path=None):
        """
        Carga el dataset desde URL o archivo local.

        Args:
            source (str): Origen de datos ('url' o 'file')
            file_path (str): Ruta del archivo si source='file'

        Returns:
            pd.DataFrame: Dataset cargado

        Raises:
            ValueError: Si el source no es válido
            FileNotFoundError: Si el archivo no existe
        """
        if source == 'url':
            url = "https://raw.githubusercontent.com/daramireh/simonBolivarCienciaDatos/refs/heads/main/Student_Performance.csv"
            try:
                df = pd.read_csv(url)
                print(f"✅ Dataset cargado desde URL: {df.shape[0]:,} filas × {df.shape[1]} columnas")
                return df
            except Exception as e:
                print(f"❌ Error al cargar datos desde URL: {e}")
                raise

        elif source == 'file':
            if file_path is None:
                file_path = '../data/Student_Performance.csv'
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Dataset cargado desde archivo: {df.shape[0]:,} filas × {df.shape[1]} columnas")
                return df
            except FileNotFoundError:
                print(f"❌ Archivo no encontrado: {file_path}")
                raise
        else:
            raise ValueError("source debe ser 'url' o 'file'")


    def validate_data(self, df):
        """
        Valida la calidad del dataset.

        Args:
            df (pd.DataFrame): Dataset a validar

        Returns:
            dict: Reporte de validación
        """
        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }

        # Verificar columnas esperadas
        expected_columns = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities',
                          'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index']

        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            validation_report['missing_columns'] = list(missing_columns)
        else:
            validation_report['missing_columns'] = []

        print("\n" + "="*70)
        print("REPORTE DE VALIDACIÓN")
        print("="*70)
        print(f"Total de registros: {validation_report['total_rows']:,}")
        print(f"Total de columnas: {validation_report['total_columns']}")
        print(f"Valores nulos: {validation_report['missing_values']}")
        print(f"Filas duplicadas: {validation_report['duplicate_rows']}")

        if validation_report['missing_columns']:
            print(f"⚠️  Columnas faltantes: {validation_report['missing_columns']}")
        else:
            print("✅ Todas las columnas esperadas están presentes")

        return validation_report


    def preprocess_data(self, df, scale_features=True):
        """
        Preprocesa el dataset completo.

        Args:
            df (pd.DataFrame): Dataset original
            scale_features (bool): Si True, escala las features numéricas

        Returns:
            pd.DataFrame: Dataset preprocesado
        """
        print("\n" + "="*70)
        print("INICIANDO PREPROCESAMIENTO")
        print("="*70)

        # Crear copia para no modificar original
        df_processed = df.copy()

        # 1. Codificar variable categórica
        print("1. Codificando variable categórica...")
        df_processed['Extracurricular Activities'] = self.label_encoder.fit_transform(
            df_processed['Extracurricular Activities']
        )
        print("   ✓ 'Extracurricular Activities' codificado (Yes=1, No=0)")

        # 2. Validar tipos de datos
        print("2. Validando tipos de datos...")
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        print(f"   ✓ {len(numeric_columns)} columnas numéricas identificadas")

        # 3. Manejar valores nulos (si existen)
        if df_processed.isnull().sum().sum() > 0:
            print("3. Manejando valores nulos...")
            df_processed = df_processed.fillna(df_processed.median(numeric_only=True))
            print("   ✓ Valores nulos imputados con mediana")
        else:
            print("3. Sin valores nulos - omitiendo imputación")

        # 4. Remover duplicados
        duplicates = df_processed.duplicated().sum()
        if duplicates > 0:
            print(f"4. Removiendo {duplicates} filas duplicadas...")
            df_processed = df_processed.drop_duplicates()
            print("   ✓ Duplicados removidos")
        else:
            print("4. Sin duplicados encontrados")

        # 5. Escalado de features (opcional)
        if scale_features:
            print("5. Escalando features numéricas...")
            features_to_scale = [col for col in numeric_columns if col != 'Performance Index']
            df_processed[features_to_scale] = self.scaler.fit_transform(df_processed[features_to_scale])
            print(f"   ✓ {len(features_to_scale)} features escaladas con StandardScaler")

        print("\n✅ PREPROCESAMIENTO COMPLETADO")
        print(f"   Dimensiones finales: {df_processed.shape[0]:,} filas × {df_processed.shape[1]} columnas")

        return df_processed


    def engineer_features(self, df):
        """
        Crea features adicionales mediante feature engineering.

        Args:
            df (pd.DataFrame): Dataset preprocesado

        Returns:
            pd.DataFrame: Dataset con features adicionales
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70)

        df_engineered = df.copy()

        # 1. Interacción: Hours Studied × Sample Question Papers
        df_engineered['Study_Practice_Interaction'] = (
            df_engineered['Hours Studied'] * df_engineered['Sample Question Papers Practiced']
        )
        print("1. ✓ Feature creado: Study_Practice_Interaction")

        # 2. Ratio: Previous Scores / Hours Studied (eficiencia)
        # Evitar división por cero
        df_engineered['Study_Efficiency'] = df_engineered.apply(
            lambda row: row['Previous Scores'] / row['Hours Studied'] if row['Hours Studied'] > 0 else 0,
            axis=1
        )
        print("2. ✓ Feature creado: Study_Efficiency")

        # 3. Feature binario: Bajo rendimiento previo
        threshold_prev_score = df_engineered['Previous Scores'].quantile(0.25)
        df_engineered['Low_Previous_Performance'] = (
            df_engineered['Previous Scores'] < threshold_prev_score
        ).astype(int)
        print(f"3. ✓ Feature creado: Low_Previous_Performance (umbral: {threshold_prev_score:.2f})")

        # 4. Feature: Total Study Effort (combinación de horas y práctica)
        df_engineered['Total_Study_Effort'] = (
            df_engineered['Hours Studied'] + df_engineered['Sample Question Papers Practiced']
        )
        print("4. ✓ Feature creado: Total_Study_Effort")

        # 5. Feature categórico: Sleep Quality
        df_engineered['Sleep_Quality'] = pd.cut(
            df_engineered['Sleep Hours'],
            bins=[0, 5, 7, 10],
            labels=['Poor', 'Moderate', 'Good']
        )
        # Codificar como numérico
        df_engineered['Sleep_Quality'] = df_engineered['Sleep_Quality'].cat.codes
        print("5. ✓ Feature creado: Sleep_Quality (Poor=0, Moderate=1, Good=2)")

        print(f"\n✅ FEATURE ENGINEERING COMPLETADO")
        print(f"   Features adicionales creados: 5")
        print(f"   Total de columnas: {df_engineered.shape[1]}")

        return df_engineered


    def split_data(self, df, target_column='Performance Index', test_size=0.2, random_state=42):
        """
        Divide el dataset en conjuntos de entrenamiento y prueba.

        Args:
            df (pd.DataFrame): Dataset procesado
            target_column (str): Nombre de la columna objetivo
            test_size (float): Proporción del conjunto de prueba
            random_state (int): Semilla para reproducibilidad

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*70)
        print("DIVISIÓN DE DATOS")
        print("="*70)

        # Separar features y target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Guardar nombres de features
        self.feature_names = X.columns.tolist()

        # División train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Total de features: {X.shape[1]}")
        print(f"Conjunto de entrenamiento: {X_train.shape[0]:,} muestras ({(1-test_size)*100:.0f}%)")
        print(f"Conjunto de prueba: {X_test.shape[0]:,} muestras ({test_size*100:.0f}%)")
        print(f"Variable objetivo: {target_column}")
        print("\n✅ División completada")

        return X_train, X_test, y_train, y_test


    def get_feature_names(self):
        """
        Retorna los nombres de las features procesadas.

        Returns:
            list: Lista de nombres de features
        """
        return self.feature_names


    def process_pipeline(self, source='url', file_path=None, engineer_features=True,
                        scale_features=True, test_size=0.2, random_state=42):
        """
        Pipeline completo de procesamiento de datos.

        Args:
            source (str): Origen de datos ('url' o 'file')
            file_path (str): Ruta del archivo si source='file'
            engineer_features (bool): Si True, aplica feature engineering
            scale_features (bool): Si True, escala las features
            test_size (float): Proporción del conjunto de prueba
            random_state (int): Semilla para reproducibilidad

        Returns:
            tuple: (X_train, X_test, y_train, y_test, df_processed)
        """
        print("="*70)
        print(" " * 15 + "PIPELINE DE PROCESAMIENTO DE DATOS")
        print("="*70)

        # 1. Cargar datos
        df = self.load_data(source=source, file_path=file_path)

        # 2. Validar datos
        self.validate_data(df)

        # 3. Preprocesar
        df_processed = self.preprocess_data(df, scale_features=scale_features)

        # 4. Feature engineering (opcional)
        if engineer_features:
            df_processed = self.engineer_features(df_processed)

        # 5. Dividir datos
        X_train, X_test, y_train, y_test = self.split_data(
            df_processed, test_size=test_size, random_state=random_state
        )

        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*70)

        return X_train, X_test, y_train, y_test, df_processed


# ====================================================================
# FUNCIÓN AUXILIAR PARA USO RÁPIDO
# ====================================================================

def quick_process(source='url', file_path=None, engineer_features=True):
    """
    Función de acceso rápido para procesar datos.

    Args:
        source (str): 'url' o 'file'
        file_path (str): Ruta del archivo local (opcional)
        engineer_features (bool): Aplicar feature engineering

    Returns:
        tuple: (X_train, X_test, y_train, y_test, processor)
    """
    processor = StudentDataProcessor()
    X_train, X_test, y_train, y_test, _ = processor.process_pipeline(
        source=source,
        file_path=file_path,
        engineer_features=engineer_features
    )

    return X_train, X_test, y_train, y_test, processor


# ====================================================================
# EJEMPLO DE USO
# ====================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 20 + "EJECUTANDO DATA PROCESSOR")
    print("="*70 + "\n")

    # Inicializar procesador
    processor = StudentDataProcessor()

    # Ejecutar pipeline completo
    X_train, X_test, y_train, y_test, df_processed = processor.process_pipeline(
        source='url',
        engineer_features=True,
        scale_features=True,
        test_size=0.2,
        random_state=42
    )

    print("\n" + "="*70)
    print("DATOS LISTOS PARA ENTRENAMIENTO")
    print("="*70)
    print(f"Features: {processor.get_feature_names()}")
