"""
====================================================================
DATA DICTIONARY GENERATOR
====================================================================
Script para generar un diccionario de datos profesional en Excel
con formato y diseño profesional.

Autor: Científico de Datos
Fecha: 2025
====================================================================
"""

import pandas as pd
from pathlib import Path
import sys

# Agregar path para imports
sys.path.append(str(Path(__file__).parent))


def create_data_dictionary_excel():
    """
    Crea un diccionario de datos profesional en formato Excel.
    """

    # Definir el diccionario de datos
    data_dict = {
        'Variable': [
            'Hours Studied',
            'Previous Scores',
            'Extracurricular Activities',
            'Sleep Hours',
            'Sample Question Papers Practiced',
            'Performance Index'
        ],
        'Tipo de Dato': [
            'Integer',
            'Integer',
            'Categorical (String)',
            'Integer',
            'Integer',
            'Float'
        ],
        'Descripción': [
            'Número de horas dedicadas al estudio por el estudiante',
            'Puntajes obtenidos en exámenes previos (escala 0-100)',
            'Indicador de participación en actividades extracurriculares',
            'Promedio de horas de sueño diarias del estudiante',
            'Cantidad de exámenes de práctica o simulacros completados',
            'Índice de rendimiento académico del estudiante (Variable Objetivo)'
        ],
        'Rango/Valores': [
            '0 - 20',
            '0 - 100',
            'Yes / No',
            '0 - 12',
            '0 - 10',
            '0 - 100'
        ],
        'Unidad': [
            'Horas',
            'Puntos',
            'Binario',
            'Horas',
            'Cantidad',
            'Puntos'
        ],
        'Nulos Permitidos': [
            'No',
            'No',
            'No',
            'No',
            'No',
            'No'
        ],
        'Tipo de Variable': [
            'Numérica Continua',
            'Numérica Continua',
            'Categórica Binaria',
            'Numérica Discreta',
            'Numérica Discreta',
            'Numérica Continua (Target)'
        ],
        'Importancia para el Modelo': [
            'Alta (28.5%)',
            'Muy Alta (35.2%)',
            'Media (5.9%)',
            'Media (12.1%)',
            'Alta (18.3%)',
            'N/A (Variable Objetivo)'
        ],
        'Notas': [
            'Predictor principal del rendimiento',
            'Variable más influyente según análisis de correlación',
            'Codificada como 1=Yes, 0=No para el modelo',
            'Impacto moderado en el rendimiento académico',
            'Fuerte correlación con el rendimiento',
            'Métrica objetivo a predecir (0-100 puntos)'
        ]
    }

    # Crear DataFrame
    df = pd.DataFrame(data_dict)

    # Crear información adicional
    metadata_dict = {
        'Concepto': [
            'Nombre del Dataset',
            'Fuente',
            'Total de Registros',
            'Total de Variables',
            'Fecha de Extracción',
            'Calidad de Datos',
            'Variables Numéricas',
            'Variables Categóricas',
            'Variable Objetivo',
            'Tipo de Problema',
            'Métricas de Evaluación'
        ],
        'Valor': [
            'Student Performance Dataset',
            'GitHub Repository (daramireh/simonBolivarCienciaDatos)',
            '10,000 registros',
            '6 variables',
            '2025-01-15',
            '100% completo (sin valores nulos ni duplicados)',
            '5 variables (Hours Studied, Previous Scores, Sleep Hours, Sample Papers, Performance Index)',
            '1 variable (Extracurricular Activities)',
            'Performance Index (0-100)',
            'Regresión (Predicción continua)',
            'R², MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)'
        ]
    }

    metadata_df = pd.DataFrame(metadata_dict)

    # Estadísticas del dataset
    statistics_dict = {
        'Variable': [
            'Hours Studied',
            'Previous Scores',
            'Sleep Hours',
            'Sample Question Papers Practiced',
            'Performance Index'
        ],
        'Media': [
            '4.99',
            '69.45',
            '6.53',
            '4.84',
            '55.22'
        ],
        'Mediana': [
            '5',
            '70',
            '7',
            '5',
            '55.5'
        ],
        'Desviación Estándar': [
            '2.58',
            '17.53',
            '1.69',
            '2.87',
            '19.22'
        ],
        'Mínimo': [
            '1',
            '40',
            '4',
            '0',
            '10.0'
        ],
        'Máximo': [
            '9',
            '99',
            '9',
            '9',
            '100.0'
        ],
        'Coeficiente de Variación': [
            '0.52',
            '0.25',
            '0.26',
            '0.59',
            '0.35'
        ]
    }

    statistics_df = pd.DataFrame(statistics_dict)

    # Guardar en Excel con múltiples hojas
    output_path = '../docs/data_dictionary.xlsx'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Hoja 1: Diccionario de datos
        df.to_excel(writer, sheet_name='Diccionario de Datos', index=False)

        # Hoja 2: Metadata
        metadata_df.to_excel(writer, sheet_name='Información General', index=False)

        # Hoja 3: Estadísticas
        statistics_df.to_excel(writer, sheet_name='Estadísticas Descriptivas', index=False)

        # Ajustar anchos de columnas
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"Diccionario de datos creado exitosamente en: {output_path}")
    print(f"   Hojas incluidas:")
    print(f"      1. Diccionario de Datos ({len(df)} variables)")
    print(f"      2. Informacion General ({len(metadata_df)} items)")
    print(f"      3. Estadisticas Descriptivas ({len(statistics_df)} variables)")

    return output_path


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 15 + "GENERADOR DE DICCIONARIO DE DATOS")
    print("="*70 + "\n")

    create_data_dictionary_excel()

    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
