"""
min_distance.py
Модуль распознавания образов на основе решающих функций минимального расстояния.
"""
from typing import List, Dict, Any, Tuple

class MinDistanceClassifier:
    def __init__(self):
        self.proto_1: List[float] = []
        self.proto_2: List[float] = []
        self.c1: float = 0.0
        self.c2: float = 0.0

    def fit(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Обучение: вычисление объектов-прототипов P1 и P2 как средних значений
        и расчет констант решающих функций D_i = 2*(X * P_i) - ||P_i||^2.
        """
        dim = len(dataset[0]["features"])
        c1_items = [o["features"] for o in dataset if o["class"] == 1]
        c2_items = [o["features"] for o in dataset if o["class"] == 2]

        # Средние прототипы
        self.proto_1 = [sum(x[d] for x in c1_items) / len(c1_items) for d in range(dim)]
        self.proto_2 = [sum(x[d] for x in c2_items) / len(c2_items) for d in range(dim)]

        # Постоянные члены: -(P_i * P_i)
        self.c1 = -sum(p ** 2 for p in self.proto_1)
        self.c2 = -sum(p ** 2 for p in self.proto_2)

    def calculate_functions(self, x: List[float]) -> Tuple[float, float]:
        """Вычисляет значения D1(X) и D2(X)."""
        d1 = 2.0 * sum(xi * pi for xi, pi in zip(x, self.proto_1)) + self.c1
        d2 = 2.0 * sum(xi * pi for xi, pi in zip(x, self.proto_2)) + self.c2
        return round(d1, 4), round(d2, 4)

    def predict(self, x: List[float]) -> int:
        """Объект относится к классу, для которого значение функции Di максимально."""
        d1, d2 = self.calculate_functions(x)
        return 1 if d1 >= d2 else 2