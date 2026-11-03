import time

import numpy as np
import matplotlib.pyplot as plt

from standard_monte_carlo import vec_standard_monte_carlo
from improved_monte_carlo import vec_improved_monte_carlo

DEF_SEED = 1234

def graficar_convergencia_N(Ns, M, S, r, sigma, B, K, x, T, prima_teorica, seed=DEF_SEED, file_name="convergencia_N.pdf"):
    primas_std = []
    primas_mejorado = []

    err_std = []
    err_mejorado = []

    for i, N in enumerate(Ns):

        np.random.seed(seed+i)
        prima_std = vec_improved_monte_carlo(M, N, S, r, sigma, B, K, x, T)

        np.random.seed(seed+i)
        prima_mejorado = vec_standard_monte_carlo(M, N, S, r, sigma, B, K, x, T)


        primas_std.append(prima_std)
        primas_mejorado.append(prima_mejorado)
        err_std.append(np.abs(prima_std-prima_teorica))
        err_mejorado.append(np.abs(prima_mejorado-prima_teorica))


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

    ax1.plot(Ns, primas_std, marker=".", label="Estándar")
    ax1.plot(Ns, primas_mejorado, marker=".", label="Mejorado")
    ax1.axhline(prima_teorica, linestyle="--", color="red", label="Valor teórico")
    ax1.set_xlabel("Número de pasos (N)")
    ax1.set_ylabel("Prima")
    ax1.legend()

    ax2.plot(Ns, err_std, marker=".", label="Estándar")
    ax2.plot(Ns, err_mejorado, marker=".", label="Mejorado")
    ax2.set_xlabel("Número de pasos (N)")
    ax2.set_ylabel("Error absoluto")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(file_name)

    return fig


M = 100000  # Número de trayectorias
N = 50      # Número de pasos de tiempo
S = 105     # Precio inicial del subyacente
r = 0.1     # Tasa libre de riesgo
sigma = 0.2 # Volatilidad del subyacente
T = 0.5     # Tiempo hasta el vencimiento
B = 100
K = 102
x = 15

Ns = np.arange(5,40,1)

fig = graficar_convergencia_N(Ns, M, S, r, sigma, B, K, x, T, 0.032311390068294976, seed=DEF_SEED, file_name="convergencia_N.pdf")
fig.show()

