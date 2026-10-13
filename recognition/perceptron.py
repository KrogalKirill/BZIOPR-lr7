"""
perceptron.py
Модуль построения разделяющей решающей функции на основе алгоритма восприятия.
"""
from typing import List, Dict, Any

class PerceptronClassifier:
    def __init__(self, max_epochs: int = 2000):
        self.max_epochs = max_epochs
        self.weights: List[float] = []
        self.w0: float = 0.0
        self.epochs_run: int = 0
        self.history: List[Dict[str, Any]] = []

    def fit(self, dataset: List[Dict[str, Any]]) -> bool:
        """
        Алгоритм восприятия:
        D12(X) = W1*X1 + ... + WN*XN + W0
        Класс 1: D12 > 0. Если D12 <= 0 -> W = W + X, W0 = W0 + 1, сброс E = 0.
        Класс 2: D12 < 0. Если D12 >= 0 -> W = W - X, W0 = W0 - 1, сброс E = 0.
        """
        dim = len(dataset[0]["features"])
        n_samples = len(dataset)

        self.weights = [0.0] * dim
        self.w0 = 0.0
        self.history = []

        e_counter = 0
        idx = 0
        total_steps = 0
        limit_steps = self.max_epochs * n_samples

        while e_counter < n_samples and total_steps < limit_steps:
            obj = dataset[idx]
            x = obj["features"]
            cls = obj["class"]

            d12 = sum(w * xi for w, xi in zip(self.weights, x)) + self.w0
            step_record = {
                "step": total_steps + 1,
                "obj_id": obj["id"],
                "class": cls,
                "d12": round(d12, 4),
                "adjusted": False
            }

            if cls == 1:
                if d12 > 0:
                    e_counter += 1
                else:
                    # Коррекция на увеличение
                    self.weights = [w + xi for w, xi in zip(self.weights, x)]
                    self.w0 += 1.0
                    e_counter = 0
                    step_record["adjusted"] = True
            else:  # Класс 2
                if d12 < 0:
                    e_counter += 1
                else:
                    # Коррекция на уменьшение
                    self.weights = [w - xi for w, xi in zip(self.weights, x)]
                    self.w0 -= 1.0
                    e_counter = 0
                    step_record["adjusted"] = True

            step_record["weights"] = [round(w, 4) for w in self.weights]
            step_record["w0"] = round(self.w0, 4)
            self.history.append(step_record)

            total_steps += 1
            idx = (idx + 1) % n_samples

        self.epochs_run = total_steps // n_samples
        return e_counter == n_samples

    def calculate_d12(self, x: List[float]) -> float:
        """Вычисляет D12(X)."""
        return round(sum(w * xi for w, xi in zip(self.weights, x)) + self.w0, 4)

    def predict(self, x: List[float]) -> int:
        """Положительное значение -> Класс 1, Отрицательное -> Класс 2."""
        d12 = self.calculate_d12(x)
        return 1 if d12 > 0 else 2