import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.stats import kstest, norm, ttest_ind
import time

N = 10000  # размер выборки

# Функция генерации выборки по ЦПТ
def generate_normal_clt(N, mean=2, sigma=1):
    # Сумма 12 равномерных чисел ~ приближение N(0,1)
    z = np.sum(np.random.rand(N, 12), axis=1) - 6
    return sigma * z + mean

# Функция генерации выборки методом Бокса–Маллера
def generate_normal_box_muller(N, mean=2, sigma=1):
    # Если N нечетное, увеличим до четного
    if N % 2 != 0:
        N += 1
    U1 = np.random.rand(N // 2)
    U2 = np.random.rand(N // 2)
    R = np.sqrt(-2 * np.log(U1))
    Theta = 2 * np.pi * U2
    Z1 = R * np.cos(Theta)
    Z2 = R * np.sin(Theta)
    # Собираем оба массива в один
    Z = np.concatenate((Z1, Z2))
    # Обрезаем до исходного размера выборки и приводим к N(2,1)
    return sigma * Z[:10000] + mean

# Измерение времени генерации и вычисление статистик для метода ЦПТ
start = time.time()
sample_clt = generate_normal_clt(N)
time_clt = time.time() - start

mean_clt = np.mean(sample_clt)
var_clt = np.var(sample_clt)

# Измерение времени генерации и вычисление статистик для метода Бокса–Маллера
start = time.time()
sample_bm = generate_normal_box_muller(N)
time_bm = time.time() - start

mean_bm = np.mean(sample_bm)
var_bm = np.var(sample_bm)

# Применяем тест Колмогорова–Смирнова для проверки соответствия N(2,1)
ks_clt = kstest(sample_clt, 'norm', args=(2, 1))
ks_bm  = kstest(sample_bm, 'norm', args=(2, 1))

# Сравниваем выборки между собой (t-тест)
t_stat, p_val = ttest_ind(sample_clt, sample_bm)

print("Метод ЦПТ:")
print("  Среднее значение: {:.4f}".format(mean_clt))
print("  Дисперсия: {:.4f}".format(var_clt))
print("  Время генерации: {:.6f} сек".format(time_clt))
print("  KS-тест: statistic = {:.4f}, p-value = {:.4f}".format(ks_clt.statistic, ks_clt.pvalue))
print()
print("Метод Бокса–Маллера:")
print("  Среднее значение: {:.4f}".format(mean_bm))
print("  Дисперсия: {:.4f}".format(var_bm))
print("  Время генерации: {:.6f} сек".format(time_bm))
print("  KS-тест: statistic = {:.4f}, p-value = {:.4f}".format(ks_bm.statistic, ks_bm.pvalue))
print()
print("t-тест (сравнение средних выборок):")
print("  t-statistic = {:.4f}, p-value = {:.4f}".format(t_stat, p_val))

# Построение гистограмм для обоих методов
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(sample_clt, bins=50, alpha=0.7, label="ЦПТ", color="blue", edgecolor='black')
plt.title("Гистограмма (ЦПТ)")
plt.xlabel("Значения")
plt.ylabel("Частота")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(sample_bm, bins=50, alpha=0.7, label="Бокс–Маллер", color="green", edgecolor='black')
plt.title("Гистограмма (Бокс–Маллер)")
plt.xlabel("Значения")
plt.ylabel("Частота")
plt.legend()

plt.tight_layout()
plt.show()
