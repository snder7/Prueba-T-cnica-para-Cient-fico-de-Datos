"""
Script de prueba para la API de predicción de rendimiento estudiantil.
"""

import requests
import json

# URL base de la API
BASE_URL = "http://localhost:8000"

def test_root():
    """Prueba el endpoint raíz"""
    print("\n" + "="*70)
    print("TEST 1: Endpoint raíz (/)")
    print("="*70)

    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def test_health():
    """Prueba el health check"""
    print("\n" + "="*70)
    print("TEST 2: Health Check (/health)")
    print("="*70)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Prueba la información del modelo"""
    print("\n" + "="*70)
    print("TEST 3: Información del Modelo (/model/info)")
    print("="*70)

    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def test_predict_single():
    """Prueba predicción individual"""
    print("\n" + "="*70)
    print("TEST 4: Predicción Individual (/predict)")
    print("="*70)

    # Estudiante con buen rendimiento esperado
    student_data = {
        "hours_studied": 7,
        "previous_scores": 85,
        "extracurricular_activities": "Yes",
        "sleep_hours": 7,
        "sample_question_papers_practiced": 5
    }

    print(f"Input:\n{json.dumps(student_data, indent=2)}")

    response = requests.post(f"{BASE_URL}/predict", json=student_data)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response:\n{json.dumps(result, indent=2)}")
        print(f"\n✓ Predicción: {result['prediction']:.2f} puntos")
        print(f"✓ Categoría: {result['prediction_category']}")
    else:
        print(f"Error:\n{json.dumps(response.json(), indent=2)}")


def test_predict_multiple_cases():
    """Prueba múltiples casos"""
    print("\n" + "="*70)
    print("TEST 5: Múltiples Casos de Prueba")
    print("="*70)

    test_cases = [
        {
            "name": "Estudiante Excelente",
            "data": {
                "hours_studied": 9,
                "previous_scores": 95,
                "extracurricular_activities": "Yes",
                "sleep_hours": 8,
                "sample_question_papers_practiced": 9
            }
        },
        {
            "name": "Estudiante Promedio",
            "data": {
                "hours_studied": 5,
                "previous_scores": 70,
                "extracurricular_activities": "No",
                "sleep_hours": 6,
                "sample_question_papers_practiced": 4
            }
        },
        {
            "name": "Estudiante en Riesgo",
            "data": {
                "hours_studied": 2,
                "previous_scores": 45,
                "extracurricular_activities": "No",
                "sleep_hours": 4,
                "sample_question_papers_practiced": 1
            }
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nCaso {i}: {test_case['name']}")
        print("-" * 70)

        response = requests.post(f"{BASE_URL}/predict", json=test_case['data'])

        if response.status_code == 200:
            result = response.json()
            print(f"  Input: {test_case['data']}")
            print(f"  Predicción: {result['prediction']:.2f} puntos")
            print(f"  Categoría: {result['prediction_category']}")
        else:
            print(f"  Error: {response.json()}")


def test_predict_batch():
    """Prueba predicción en lote"""
    print("\n" + "="*70)
    print("TEST 6: Predicción en Lote (/predict/batch)")
    print("="*70)

    batch_data = [
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
        },
        {
            "hours_studied": 9,
            "previous_scores": 95,
            "extracurricular_activities": "Yes",
            "sleep_hours": 8,
            "sample_question_papers_practiced": 8
        }
    ]

    print(f"Input: {len(batch_data)} estudiantes")

    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response:\n{json.dumps(result, indent=2)}")
        print(f"\n✓ Total predicciones: {result['total_predictions']}")
        for i, (pred, cat) in enumerate(zip(result['predictions'], result['prediction_categories']), 1):
            print(f"  Estudiante {i}: {pred:.2f} puntos - {cat}")
    else:
        print(f"Error:\n{json.dumps(response.json(), indent=2)}")


def main():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*70)
    print(" " * 15 + "SUITE DE PRUEBAS DE LA API")
    print("="*70)
    print(f"URL Base: {BASE_URL}")
    print("Asegúrate de que la API esté corriendo en http://localhost:8000")
    print("="*70)

    try:
        # Ejecutar todas las pruebas
        test_root()
        test_health()
        test_model_info()
        test_predict_single()
        test_predict_multiple_cases()
        test_predict_batch()

        print("\n" + "="*70)
        print("✓ TODAS LAS PRUEBAS COMPLETADAS")
        print("="*70 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n" + "="*70)
        print("ERROR: No se pudo conectar a la API")
        print("="*70)
        print("Asegúrate de que la API esté corriendo:")
        print("  cd src")
        print("  python api.py")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\nError inesperado: {e}")


if __name__ == "__main__":
    main()
