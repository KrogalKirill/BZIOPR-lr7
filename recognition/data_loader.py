"""
data_loader.py
Модуль формирования обучающего множества и нормализации данных.
Включает 12 объектов (2 класса по 6 объектов) и 5 признаков (N >= 10, M >= 5).
"""
from typing import List, Dict, Any, Tuple

def get_training_dataset() -> List[Dict[str, Any]]:
    """
    Обучающее множество объектов технической диагностики серверных узлов:
    Класс 1: Нормальный режим работы
    Класс 2: Предаварийный режим / повышенная нагрузка

    Признаки:
      - X1: Загрузка CPU (%)
      - X2: Использование RAM (%)
      - X3: Температура кристалла (°C)
      - X4: Число сетевых ошибок (ош/мин)
      - X5: Задержка дисковой подсистемы (мс)
    """
    raw_data = [
        # Класс 1 (Норма)
        {"id": "Obj_01", "class": 1, "features": [25.0, 40.0, 45.0,  2.0, 10.0]},
        {"id": "Obj_02", "class": 1, "features": [30.0, 45.0, 50.0,  5.0, 12.0]},
        {"id": "Obj_03", "class": 1, "features": [20.0, 35.0, 42.0,  1.0,  8.0]},
        {"id": "Obj_04", "class": 1, "features": [35.0, 50.0, 52.0,  4.0, 15.0]},
        {"id": "Obj_05", "class": 1, "features": [28.0, 42.0, 48.0,  3.0, 11.0]},
        {"id": "Obj_06", "class": 1, "features": [22.0, 38.0, 44.0,  2.0,  9.0]},

        # Класс 2 (Предаварийный)
        {"id": "Obj_07", "class": 2, "features": [85.0, 88.0, 78.0, 45.0, 65.0]},
        {"id": "Obj_08", "class": 2, "features": [90.0, 92.0, 82.0, 60.0, 70.0]},
        {"id": "Obj_09", "class": 2, "features": [78.0, 80.0, 75.0, 38.0, 55.0]},
        {"id": "Obj_10", "class": 2, "features": [95.0, 96.0, 85.0, 72.0, 80.0]},
        {"id": "Obj_11", "class": 2, "features": [82.0, 85.0, 76.0, 40.0, 62.0]},
        {"id": "Obj_12", "class": 2, "features": [88.0, 90.0, 80.0, 52.0, 68.0]},
    ]
    return raw_data

def get_test_objects() -> List[Dict[str, Any]]:
    """Тестовые контрольные объекты для проверки работы решающих функций."""
    return [
        {"id": "Test_A", "true_class": 1, "features": [27.0, 41.0, 46.0,  3.0, 10.0]},
        {"id": "Test_B", "true_class": 2, "features": [84.0, 87.0, 79.0, 48.0, 64.0]},
        {"id": "Test_C", "true_class": 1, "features": [32.0, 48.0, 51.0,  6.0, 14.0]},
    ]

def normalize_dataset(
    dataset: List[Dict[str, Any]],
    max_vals: List[float] = None
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Приведение признаков к безразмерному виду путем деления на максимум по признаку.
    """
    dim = len(dataset[0]["features"])
    if max_vals is None:
        max_vals = [0.0] * dim
        for obj in dataset:
            for d in range(dim):
                if obj["features"][d] > max_vals[d]:
                    max_vals[d] = obj["features"][d]

    normalized = []
    for obj in dataset:
        norm_feat = [round(obj["features"][d] / max_vals[d], 4) for d in range(dim)]
        item = {
            "id": obj["id"],
            "raw_features": obj["features"],
            "features": norm_feat
        }
        if "class" in obj:
            item["class"] = obj["class"]
        if "true_class" in obj:
            item["true_class"] = obj["true_class"]
        normalized.append(item)

    return normalized, max_vals