import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox

class PolynomialApp:
    def __init__(self, master):
        self.master = master
        master.title("Инструмент аппроксимации полинома")

        self.defaults = [
            [-1, -2],
            [0, -2],
            [1, -7],
            [2, 1],
            [3, 14]
        ]
        self.data_points = [tuple(point) for point in self.defaults]
        self.plotted_polynomials = set() # To keep track of plotted polynomials

        # --- Область отображения данных ---
        self.data_frame = ttk.LabelFrame(master, text="Входные данные (x, f(x))")
        self.data_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.data_table = ttk.Treeview(self.data_frame, columns=("x", "f(x)"), show="headings")
        self.data_table.heading("x", text="x")
        self.data_table.heading("f(x)", text="f(x)")
        self.data_table.pack(padx=5, pady=5, fill="both", expand=True)

        # Заполнить таблицу данных значениями по умолчанию
        for x, y in self.data_points:
            self.data_table.insert("", tk.END, values=(x, y))

        # --- Область выбора и расчета полиномов ---
        self.poly_frame = ttk.LabelFrame(master, text="Выберите тип полинома")
        self.poly_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ttk.Button(self.poly_frame, text="Интерполяция Ньютона", command=self.plot_newton).pack(pady=5, padx=10, fill="x")
        ttk.Button(self.poly_frame, text="Интерполяция Лагранжа", command=self.plot_lagrange).pack(pady=5, padx=10, fill="x")

        self.ls_frame = ttk.LabelFrame(self.poly_frame, text="Сглаживание методом наименьших квадратов")
        self.ls_frame.pack(pady=5, padx=10, fill="x")
        self.ls_degree = tk.IntVar(value=1)
        ttk.Radiobutton(self.ls_frame, text="Степень 1", variable=self.ls_degree, value=1).pack(anchor="w")
        ttk.Radiobutton(self.ls_frame, text="Степень 2", variable=self.ls_degree, value=2).pack(anchor="w")
        ttk.Radiobutton(self.ls_frame, text="Степень 3", variable=self.ls_degree, value=3).pack(anchor="w")
        ttk.Button(self.ls_frame, text="Рассчитать и построить", command=self.plot_least_squares).pack(pady=5)

        self.fourth_deg_frame = ttk.LabelFrame(self.poly_frame, text="Произвольный полином 4-й степени")
        self.fourth_deg_frame.pack(pady=5, padx=10, fill="x")
        self.coeff_entries = []
        for i in range(5):
            ttk.Label(self.fourth_deg_frame, text=f"Коэф x^{i}:").pack(anchor="w")
            entry = ttk.Entry(self.fourth_deg_frame, width=15)
            entry.pack(anchor="w")
            self.coeff_entries.append(entry)
        # Установить коэффициенты по умолчанию
        default_coeffs = ["-2.0", "0.5", "-3.0", "1.5", "0.0"]
        for i, coeff in enumerate(default_coeffs):
            self.coeff_entries[i].insert(0, coeff)
        ttk.Button(self.fourth_deg_frame, text="Построить полином", command=self.plot_fourth_degree).pack(pady=5)

        ttk.Button(self.poly_frame, text="Очистить график", command=self.clear_plot).pack(pady=10, padx=10, fill="x")

        # --- Область графика ---
        self.plot_frame = ttk.LabelFrame(master, text="График полиномов")
        self.plot_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        self.ax.grid(True)

        # Построить график с данными по умолчанию при запуске
        if self.data_points:
            x_data, y_data = zip(*self.data_points)
            self.ax.scatter(x_data, y_data, color='red', label='Исходные точки данных')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('f(x)')
            self.ax.set_title('Аппроксимация полинома')
            self.ax.legend()
            self.canvas.draw()

        # Настройка расположения элементов
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(1, weight=1)

    def get_plot_points(self, func, x_data):
        if not x_data:
            return [], []
        x_min = min(x_data)
        x_max = max(x_data)
        x_plot = np.linspace(x_min, x_max, 100) # Шаг ~ (max-min)/100, можно настроить
        y_plot = [func(xi) for xi in x_plot]
        return x_plot, y_plot

    def plot_polynomial(self, poly_x, poly_y, label):
        if label in self.plotted_polynomials:
            # Удалить существующий график с тем же именем
            for line in self.ax.lines:
                if line.get_label() == label:
                    line.remove()
        self.ax.plot(poly_x, poly_y, label=label)
        self.plotted_polynomials.add(label)
        self.ax.legend()
        self.canvas.draw()

    def lagrange_interpolation(self, x, x_data, y_data):
        n = len(x_data)
        result = 0.0
        for i in range(n):
            term = y_data[i]
            for j in range(n):
                if i != j:
                    term *= (x - x_data[j]) / (x_data[i] - x_data[j])
            result += term
        return result

    def plot_lagrange(self):
        if len(self.data_points) < 2:
            messagebox.showerror("Ошибка", "Для интерполяции Лагранжа требуется как минимум две точки.")
            return
        label = "Интерполяция Лагранжа"
        if label in self.plotted_polynomials:
            for line in self.ax.lines:
                if line.get_label() == label:
                    line.remove()
        x_data, y_data = zip(*self.data_points)
        lagrange_func = lambda x: self.lagrange_interpolation(x, x_data, y_data)
        poly_x, poly_y = self.get_plot_points(lagrange_func, x_data)
        self.plot_polynomial(poly_x, poly_y, label)

    def divided_differences(self, x_data, y_data):
        n = len(y_data)
        dd_table = np.zeros((n, n))
        dd_table[:, 0] = y_data
        for j in range(1, n):
            for i in range(n - j):
                dd_table[i, j] = (dd_table[i + 1, j - 1] - dd_table[i, j - 1]) / (x_data[i + j] - x_data[i])
        return dd_table

    def newton_interpolation(self, x, x_data, y_data):
        dd_table = self.divided_differences(x_data, y_data)
        n = len(x_data)
        result = dd_table[0, 0]
        term = 1.0
        for i in range(1, n):
            term *= (x - x_data[i - 1])
            result += dd_table[0, i] * term
        return result

    def plot_newton(self):
        if len(self.data_points) < 2:
            messagebox.showerror("Ошибка", "Для интерполяции Ньютона требуется как минимум две точки.")
            return
        label = "Интерполяция Ньютона"
        if label in self.plotted_polynomials:
            for line in self.ax.lines:
                if line.get_label() == label:
                    line.remove()
        x_data, y_data = zip(*self.data_points)
        newton_func = lambda x: self.newton_interpolation(x, x_data, y_data)
        poly_x, poly_y = self.get_plot_points(newton_func, x_data)
        self.plot_polynomial(poly_x, poly_y, label)

    def least_squares_polynomial(self, x_data, y_data, degree):
        A = np.vander(x_data, degree + 1)
        coeffs, residuals, rank, s = np.linalg.lstsq(A, y_data, rcond=None)
        return coeffs[::-1]

    def plot_least_squares(self):
        if len(self.data_points) < self.ls_degree.get() + 1:
            messagebox.showerror("Ошибка", f"Для метода наименьших квадратов степени {self.ls_degree.get()} требуется как минимум {self.ls_degree.get() + 1} точек.")
            return
        degree = self.ls_degree.get()
        label = f"Сглаживание МНК (степень {degree})"
        if label in self.plotted_polynomials:
            for line in self.ax.lines:
                if line.get_label() == label:
                    line.remove()
        x_data, y_data = zip(*self.data_points)
        coeffs = self.least_squares_polynomial(x_data, y_data, degree)
        poly = np.poly1d(coeffs)
        poly_x, poly_y = self.get_plot_points(poly, x_data)
        self.plot_polynomial(poly_x, poly_y, label)
        print(f"Коэффициенты МНК (степень {degree}): {coeffs}")

    def fourth_degree_polynomial(self, x, coeffs):
        return coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * x**3 + coeffs[4] * x**4

    def plot_fourth_degree(self):
        try:
            coeffs_str = [entry.get() for entry in self.coeff_entries]
            coeffs = [float(coeff) for coeff in coeffs_str]
            if len(coeffs) != 5:
                raise ValueError("Пожалуйста, введите 5 коэффициентов.")
            label = "Произвольный полином 4-й степени"
            if label in self.plotted_polynomials:
                for line in self.ax.lines:
                    if line.get_label() == label:
                        line.remove()
            poly_func = lambda x: self.fourth_degree_polynomial(x, coeffs)
            if self.data_points:
                x_data, _ = zip(*self.data_points)
                poly_x, poly_y = self.get_plot_points(poly_func, x_data)
            else:
                poly_x = np.linspace(-5, 5, 100) # Диапазон по умолчанию, если нет данных
                poly_y = [poly_func(xi) for xi in poly_x]
            self.plot_polynomial(poly_x, poly_y, label)
            print(f"Коэффициенты полинома 4-й степени: {coeffs}")
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))

    def clear_plot(self):
        for line in self.ax.lines:
            if line.get_label() != 'Исходные точки данных':
                line.remove()
        self.plotted_polynomials.clear()
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PolynomialApp(root)
    root.mainloop()