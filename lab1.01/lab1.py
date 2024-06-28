import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
from math import sqrt, pi, exp


def find_acc(x):
    xi = ""
    t = False
    for e in str(x):
        if not t:
            t = e in "123456789"
        if t:
            xi += e
    r = str(x).find(xi)
    if xi[0] in "123":
        return round(x, r), r
    return round(x, r - 1), r - 1


def smooth_curve(T, power):
    xnew = np.linspace(T.min(), T.max(), 300) 
    spl = make_interp_spline(T, power, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return xnew, power_smooth


data = np.array(list(map(float, open("data.txt").readlines())), dtype=np.float64)
N = len(data)

# Шаг 1
t_min, t_max = min(data), max(data)
m = 7 # round(sqrt(50))
delta_t = (t_max - t_min) / m
# delta_t = np.round(delta_t, 2)
# print(delta_t)
intervals = [(t_min + e * delta_t, t_max - (m - 1 - e) * delta_t) for e in  range(m)]
# intervals = [(4.9, 4.93), (4.93, 4.96), (4.96, 4.99), (4.99, 5.03), (5.03, 5.06), (5.06, 5.09), (5.09, 5.12)]
# print(intervals)
# intervals = [(round(i[0], 3), round(i[1], 3)) for i in intervals]
# print(intervals)

values_in_intervals = [data[(t_l <= data) & (data <= t_r)] for t_l, t_r in intervals]
values_in_intervals[-1] = np.append(values_in_intervals[-1], t_max)
delta_N_values = list(map(len, values_in_intervals))
# print(*delta_N_values, sep="\n")
experimental_density_values = list(map(lambda x: x / (N * delta_t), delta_N_values))
# print(*np.round(experimental_density_values, 2).tolist(), sep="\n")
# Шаг 2 и 3
mean_t = data.mean()
diff = data - mean_t
# print(sum(diff))
summ_diff = np.sum(diff)
diff_sq = diff ** 2
sigma_n = sqrt(sum(diff_sq) / (N - 1))

# Шаг 4
rho_max = 1 / (sigma_n * sqrt(2 * pi))
# Шаг 5
mean_interval_values = list(map(lambda x: (x[0] + x[1]) / 2, intervals))
# print([round(i, 3) for i in mean_interval_values])
p_t = lambda t: 1 / (sigma_n * sqrt(2 * pi)) * exp(-(t - mean_t) ** 2 / (2 * sigma_n ** 2))  # noqa: E731
rho_values = list(map(p_t, mean_interval_values))
# print(*np.round(rho_values, 3).tolist(), sep="\n")
# Шаг 6
sigma_intervals = [(mean_t - e * sigma_n, mean_t + e * sigma_n) for e in range(1, 4)]
# print([(round(i[0], 3), round(i[1], 3)) for i in sigma_intervals])
# Шаг 7
values_in_sigma_intervals = [data[(t_l <= data) & (data <= t_r)] for t_l, t_r in sigma_intervals]
sigma_delta_N_values = list(map(len, values_in_sigma_intervals))
P_values = [0.683, 0.954, 0.997]
experimental_P_values = list(map(lambda x: x / N, sigma_delta_N_values))
# print(experimental_P_values)
# Шаг 8
sigma_mean = sqrt(sum(diff_sq) / (N * (N - 1)))
print(sigma_mean)
print("A", np.sqrt(sigma_mean ** 2 + (2 * 0.005 / 3) ** 2))
# Шаг 9
student_coeff = 2.009575234489209 # https://www.kontrolnaya-rabota.ru/s/teoriya-veroyatnosti/tablica-studenta/?n=50&p=0.95
# print(round(student_coeff, 5))
trust_delta_t = student_coeff * sigma_mean
# print(trust_delta_t)
print(mean_t)
trust_interval = (mean_t - trust_delta_t, mean_t + trust_delta_t)
random_error = trust_delta_t
measure_device_error = (1/ 100) / 2

abs_error_directed = np.sqrt(random_error ** 2 + (2/3 * measure_device_error) ** 2)
# print(abs_error_directed / mean_t * 100)
# print(trust_interval)

# Гистограмма
plt.bar(mean_interval_values, experimental_density_values, width=delta_t, alpha=0.7, linewidth=1.3, edgecolor="black", color="white")
plt.plot(*smooth_curve(np.array(mean_interval_values), np.array(rho_values)), color='black')
plt.grid(linestyle="--")
plt.xlabel(r"$t$", fontsize=14)
plt.ylabel(r"$\frac{\Delta N}{N\Delta t}$", rotation=0, labelpad=20, fontsize=14)
plt.savefig("histogram.png")
# print(experimental_density_values)
# print(rho_values)
