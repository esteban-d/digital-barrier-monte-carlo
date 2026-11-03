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
    ax1.title.set_text("Prima")
    ax1.legend()

    ax2.plot(Ns, err_std, marker=".", label="Estándar")
    ax2.plot(Ns, err_mejorado, marker=".", label="Mejorado")
    ax2.set_xlabel("Número de pasos (N)")
    ax2.set_ylabel("Error absoluto")
    ax2.title.set_text("Error")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(file_name)

    return fig


M = 100000  # Número de trayectorias
N = 40      # Número de pasos de tiempo
S = 105     # Precio inicial del subyacente
r = 0.1     # Tasa libre de riesgo
sigma = 0.2 # Volatilidad del subyacente
T = 0.5     # Tiempo hasta el vencimiento
B = 100
K = 102
x = 15





def graficar_convergencia_M(Ms, N, S, r, sigma, B, K, x, T, prima_teorica, n_reps=30, seed=DEF_SEED, file_name="convergencia_M.pdf"):

    err_std = np.empty((len(Ms), n_reps))
    err_mej = np.empty((len(Ms), n_reps))

    for i, M in enumerate(Ms):

        for rep in range(n_reps):

            iter_seed = seed + rep

            np.random.seed(iter_seed)
            prima_std = vec_standard_monte_carlo(M, N, S, r, sigma, B, K, x, T)

            np.random.seed(iter_seed)
            prima_mejorado = vec_improved_monte_carlo(M, N, S, r, sigma, B, K, x, T)

            err_std[i, rep] = np.abs(prima_std-prima_teorica)
            err_mej[i, rep] = np.abs(prima_mejorado-prima_teorica)
        
    fig, ax = plt.subplots(figsize=(10, 7))

    for err, label in [(err_std, "Estándar"), (err_mej, "Mejorado")]:
        media = err.mean(axis=1)
        p25, p75 = np.percentile(err, [25, 75], axis=1)
        color = ax.plot(Ms, media, marker=".", label=label)[0].get_color()
        ax.fill_between(Ms, p25, p75, alpha=0.3, color=color)

    media_mej = err_mej.mean(axis=1)
    c = media_mej[0] * np.sqrt(Ms[0])
    ax.plot(Ms, c/np.sqrt(Ms), "k--", label=r"$\propto 1/\sqrt{M}$")

    ax.set_xlabel("Número de simulaciones (M)")
    ax.set_ylabel("Error absoluto")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.title.set_text(f"Error (N={N})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(file_name)

    return fig




def graficar_tiempos_ejecucion(Ms, N, S, r, sigma, B, K, x, T, prima_teorica, n_reps=10, seed=DEF_SEED, file_name="convergencia_tiempos_ejecucion.pdf"):
    metodos = {
        "Estandar": vec_standard_monte_carlo,
        "Mejorado": vec_improved_monte_carlo
    }

    tiempos = {nombre: [] for nombre in metodos}

    for nombre, metodo in metodos.items():
        for M in Ms:
            tiempos_M = []

            for repeticion in range(n_reps):
                np.random.seed(seed+repeticion)

                inicio = time.perf_counter()
                metodo(M, N, S, r, sigma, B, K, x, T)
                tiempos_M.append(time.perf_counter() - inicio)

            tiempos[nombre].append(np.mean(tiempos_M))
    
    fig, ax = plt.subplots(figsize=(7, 5))

    for nombre, valores in tiempos.items():
        ax.plot(Ms, valores, marker="o", label=nombre)

    ax.set_xlabel("Número de simulaciones (M)")
    ax.set_ylabel("Tiempo mediano (segundos)")
    ax.legend()

    fig.tight_layout()

    fig.savefig(file_name)
    
    return fig

