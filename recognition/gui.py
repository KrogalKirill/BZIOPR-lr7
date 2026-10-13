"""
gui.py
Графический интерфейс пользователя (Tkinter/Canvas).
"""
import tkinter as tk
from tkinter import ttk, messagebox
from data_loader import get_training_dataset, get_test_objects, normalize_dataset
from min_distance import MinDistanceClassifier
from perceptron import PerceptronClassifier

class RecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР №7 — Методы распознавания")
        self.geometry("1100x750")
        self.minsize(950, 650)

        # Подготовка обучающей выборки и нормализации
        self.raw_train = get_training_dataset()
        self.train_norm, self.max_vals = normalize_dataset(self.raw_train)
        self.test_norm, _ = normalize_dataset(get_test_objects(), self.max_vals)

        # Модели
        self.min_dist_model = MinDistanceClassifier()
        self.perceptron_model = PerceptronClassifier()

        self._build_ui()
        self._populate_training_table()
        self.train_models()

    def _build_ui(self):
        # Верхняя панель управления и распознавания тестового объекта
        top_frame = ttk.LabelFrame(self, text="Входные данные распознаваемого объекта (не нормализованные)")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.feature_entries = []
        labels = ["X1 (CPU %):", "X2 (RAM %):", "X3 (Temp °C):", "X4 (Ошибки):", "X5 (Задержка):"]
        default_vals = ["27.0", "41.0", "46.0", "3.0", "10.0"]

        for col, (lbl, val) in enumerate(zip(labels, default_vals)):
            ttk.Label(top_frame, text=lbl).grid(row=0, column=col * 2, padx=4, pady=5, sticky=tk.W)
            ent = ttk.Entry(top_frame, width=7)
            ent.insert(0, val)
            ent.grid(row=0, column=col * 2 + 1, padx=4, pady=5)
            self.feature_entries.append(ent)

        btn_recognize = ttk.Button(top_frame, text="Распознать объект", command=self.recognize_custom)
        btn_recognize.grid(row=0, column=10, padx=10, pady=5)

        btn_retrain = ttk.Button(top_frame, text="Переобучить модели", command=self.train_models)
        btn_retrain.grid(row=0, column=11, padx=5, pady=5)

        # Разделяемая область
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Левый фрейм: Таблицы и протокол
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        ttk.Label(left_frame, text="Обучающая выборка (12 объектов, 5 признаков, 2 класса):").pack(anchor=tk.W, pady=2)
        cols = ("id", "cls", "x1", "x2", "x3", "x4", "x5")
        self.tree = ttk.Treeview(left_frame, columns=cols, show="headings", height=7)
        for c in cols:
            self.tree.column(c, width=65, anchor=tk.CENTER)
        self.tree.heading("id", text="ID")
        self.tree.heading("cls", text="Класс")
        self.tree.heading("x1", text="X1 (CPU)")
        self.tree.heading("x2", text="X2 (RAM)")
        self.tree.heading("x3", text="X3 (Temp)")
        self.tree.heading("x4", text="X4 (Net)")
        self.tree.heading("x5", text="X5 (Disk)")
        self.tree.pack(fill=tk.X, pady=2)

        ttk.Label(left_frame, text="Протокол работы алгоритмов и решающих функций:").pack(anchor=tk.W, pady=2)
        self.log_text = tk.Text(left_frame, wrap=tk.WORD, height=16)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=2)

        # Правый фрейм: Визуализация геометрической интерпретации (X1 vs X2)
        right_frame = ttk.LabelFrame(paned, text="Геометрическая интерпретация (Проекция X1 - X2)")
        paned.add(right_frame, weight=2)

        self.canvas = tk.Canvas(right_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _populate_training_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for obj in self.train_norm:
            f = obj["features"]
            self.tree.insert("", tk.END, values=(obj["id"], f"Класс {obj['class']}", f[0], f[1], f[2], f[3], f[4]))

    def train_models(self):
        # 1. Обучение минимального расстояния
        self.min_dist_model.fit(self.train_norm)
        # 2. Обучение алгоритма восприятия
        success = self.perceptron_model.fit(self.train_norm)

        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "=== 1. МЕТОД МИНИМАЛЬНОГО РАССТОЯНИЯ ===\n")
        p1 = [round(v, 4) for v in self.min_dist_model.proto_1]
        p2 = [round(v, 4) for v in self.min_dist_model.proto_2]
        self.log_text.insert(tk.END, f"Прототип P1 (Класс 1): {p1}\n")
        self.log_text.insert(tk.END, f"Прототип P2 (Класс 2): {p2}\n")
        self.log_text.insert(tk.END, f"Константы функций: C1 = {round(self.min_dist_model.c1, 4)}, C2 = {round(self.min_dist_model.c2, 4)}\n\n")

        self.log_text.insert(tk.END, "=== 2. АЛГОРИТМ ВОСПРИЯТИЯ (ПЕРЦЕПТРОН) ===\n")
        w = [round(v, 4) for v in self.perceptron_model.weights]
        w0 = round(self.perceptron_model.w0, 4)
        self.log_text.insert(tk.END, f"Статус сходимости: {'Успешно' if success else 'Превышен лимит'}\n")
        self.log_text.insert(tk.END, f"Коэффициенты гиперплоскости: W = {w}, W0 = {w0}\n")
        self.log_text.insert(tk.END, f"Уравнение: D12 = {' + '.join(f'{wi}*X{i+1}' for i, wi in enumerate(w))} + ({w0})\n\n")

        # Тестирование предопределенных объектов
        self.log_text.insert(tk.END, "=== РАСПОЗНАВАНИЕ КОНТРОЛЬНЫХ ТЕСТОВЫХ ОБЪЕКТОВ ===\n")
        for t in self.test_norm:
            d1, d2 = self.min_dist_model.calculate_functions(t["features"])
            pred_md = self.min_dist_model.predict(t["features"])
            d12 = self.perceptron_model.calculate_d12(t["features"])
            pred_pt = self.perceptron_model.predict(t["features"])
            self.log_text.insert(
                tk.END,
                f"[{t['id']}] Истинный: {t['true_class']} | "
                f"Мин. расст: Класс {pred_md} (D1={d1}, D2={d2}) | "
                f"Восприятие: Класс {pred_pt} (D12={d12})\n"
            )

        self._draw_2d_plot()

    def recognize_custom(self):
        try:
            raw_vals = [float(ent.get().strip()) for ent in self.feature_entries]
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Все признаки должны быть вещественными числами!")
            return

        # Нормализация введенного объекта
        norm_vals = [round(raw_vals[i] / self.max_vals[i], 4) for i in range(len(raw_vals))]

        d1, d2 = self.min_dist_model.calculate_functions(norm_vals)
        pred_md = self.min_dist_model.predict(norm_vals)

        d12 = self.perceptron_model.calculate_d12(norm_vals)
        pred_pt = self.perceptron_model.predict(norm_vals)

        self.log_text.insert(tk.END, "\n--- РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ ПОЛЬЗОВАТЕЛЬСКОГО ОБЪЕКТА ---\n")
        self.log_text.insert(tk.END, f"Входной вектор (сырой): {raw_vals}\n")
        self.log_text.insert(tk.END, f"Нормализованный вектор: {norm_vals}\n")
        self.log_text.insert(tk.END, f"1) Метод мин. расстояния: Класс {pred_md} (D1 = {d1}, D2 = {d2})\n")
        self.log_text.insert(tk.END, f"2) Алгоритм восприятия:   Класс {pred_pt} (D12 = {d12})\n")
        self.log_text.see(tk.END)

        self._draw_2d_plot(custom_point=norm_vals)

    def _draw_2d_plot(self, custom_point=None):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 400
        h = self.canvas.winfo_height() or 400
        pad = 45

        # Оси координат
        self.canvas.create_line(pad, h - pad, w - pad, h - pad, arrow=tk.LAST, fill="#444", width=2)
        self.canvas.create_line(pad, h - pad, pad, pad, arrow=tk.LAST, fill="#444", width=2)
        self.canvas.create_text(w - pad, h - pad + 15, text="X1 (CPU)", fill="#222")
        self.canvas.create_text(pad - 20, pad, text="X2 (RAM)", fill="#222")

        # Обучающие точки
        for obj in self.train_norm:
            x_val, y_val = obj["features"][0], obj["features"][1]
            cx = pad + x_val * (w - 2 * pad)
            cy = (h - pad) - y_val * (h - 2 * pad)
            color = "#2980b9" if obj["class"] == 1 else "#e74c3c"
            self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=color, outline="#111")
            self.canvas.create_text(cx + 8, cy - 6, text=obj["id"], font=("Arial", 8))

        # Прототипы P1 и P2
        for idx, proto in enumerate([self.min_dist_model.proto_1, self.min_dist_model.proto_2]):
            px = pad + proto[0] * (w - 2 * pad)
            py = (h - pad) - proto[1] * (h - 2 * pad)
            self.canvas.create_line(px - 7, py - 7, px + 7, py + 7, width=2, fill="#27ae60")
            self.canvas.create_line(px - 7, py + 7, px + 7, py - 7, width=2, fill="#27ae60")
            self.canvas.create_text(px, py - 12, text=f"P{idx+1}", fill="#27ae60", font=("Arial", 9, "bold"))

        # Разделяющая граница алгоритма восприятия в проекции (W1*X1 + W2*X2 + W0 = 0)
        w1 = self.perceptron_model.weights[0] if len(self.perceptron_model.weights) > 0 else 0
        w2 = self.perceptron_model.weights[1] if len(self.perceptron_model.weights) > 1 else 0
        w0 = self.perceptron_model.w0
        if abs(w2) > 1e-5:
            # Находим точки при X1 = 0 и X1 = 1
            y_at_0 = -w0 / w2
            y_at_1 = -(w0 + w1) / w2
            x_c1, y_c1 = pad, (h - pad) - y_at_0 * (h - 2 * pad)
            x_c2, y_c2 = pad + (w - 2 * pad), (h - pad) - y_at_1 * (h - 2 * pad)
            self.canvas.create_line(x_c1, y_c1, x_c2, y_c2, fill="#8e44ad", width=2, dash=(4, 2))
            self.canvas.create_text(w - pad - 40, pad + 15, text="Граница D12=0", fill="#8e44ad", font=("Arial", 8, "italic"))

        # Точка распознаваемого объекта
        if custom_point:
            cx = pad + custom_point[0] * (w - 2 * pad)
            cy = (h - pad) - custom_point[1] * (h - 2 * pad)
            self.canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill="#f1c40f", outline="black", width=2)
            self.canvas.create_text(cx, cy - 14, text="X_тест", fill="black", font=("Arial", 9, "bold"))