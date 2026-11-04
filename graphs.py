import time
import numpy as np
import matplotlib.pyplot as plt

from standard_monte_carlo import vec_standard_monte_carlo
from improved_monte_carlo import vec_improved_monte_carlo
from pathlib import Path


def graficar_convergencia_N(Ns, M, S, r, sigma, B, K, x, T, prima_teorica, file_name="results/convergencia_N.pdf"):
    primas_std = []
    primas_mejorado = []

    err_std = []
    err_mejorado = []

    for i, N in enumerate(Ns):

        print(f"Graficando convergencia respecto a N: Simulando N={N}.")

        prima_std = vec_standard_monte_carlo(M, N, S, r, sigma, B, K, x, T)

        prima_mejorado = vec_improved_monte_carlo(
            M, N, S, r, sigma, B, K, x, T)

        primas_std.append(prima_std)
        primas_mejorado.append(prima_mejorado)
        err_std.append(np.abs(prima_std-prima_teorica))
        err_mejorado.append(np.abs(prima_mejorado-prima_teorica))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

    ax1.plot(Ns, primas_std, marker=".", label="Estándar")
    ax1.plot(Ns, primas_mejorado, marker=".", label="Mejorado")
    ax1.axhline(prima_teorica, linestyle="--",
                color="red", label="Valor teórico")
    ax1.set_xlabel("Número de pasos (N)")
    ax1.set_ylabel("Prima")
    ax1.title.set_text("Prima")
    ax1.set_xscale("log")
    #ax1.set_yscale("log")
    ax1.legend()

    c_N = err_std[0] * np.sqrt(Ns[0])
    ax2.plot(Ns, c_N / np.sqrt(Ns), "k--", label=r"Referencia $1/\sqrt{N}$")

    ax2.plot(Ns, err_std, marker=".", label="Estándar")
    ax2.plot(Ns, err_mejorado, marker=".", label="Mejorado")
    ax2.set_xlabel("Número de pasos (N)")
    ax2.set_ylabel("Error absoluto")
    ax2.title.set_text("Error")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend()


    fig.tight_layout()

    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_name)

    return fig


def graficar_convergencia_M(Ms, N, S, r, sigma, B, K, x, T, prima_teorica, n_reps=30, file_name="results/convergencia_M.pdf"):

    err_std = np.empty((len(Ms), n_reps))
    err_mej = np.empty((len(Ms), n_reps))

    for i, M in enumerate(Ms):

        for rep in range(n_reps):

            print(f"Graficando convergencia M: Simulando M={M}, repeticion {rep}.")

            prima_std = vec_standard_monte_carlo(M, N, S, r, sigma, B, K, x, T)

            prima_mejorado = vec_improved_monte_carlo(
                M, N, S, r, sigma, B, K, x, T)

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
    ax.plot(Ms, c/np.sqrt(Ms), "k--", label=r"Referencia $1/\sqrt{M}$")

    ax.title.set_text(f"Error (N={N})")
    ax.set_xlabel("Número de simulaciones (M)")
    ax.set_ylabel("Error absoluto")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()

    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_name)

    return fig


def graficar_tiempos_ejecucion_M(Ms, N, S, r, sigma, B, K, x, T, n_reps=10 , file_name="results/convergencia_tiempos_ejecucion_M.pdf"):
    metodos = {
        "Estándar": vec_standard_monte_carlo,
        "Mejorado": vec_improved_monte_carlo
    }

    tiempos = {nombre: [] for nombre in metodos}

    for nombre, metodo in metodos.items():
        for M in Ms:
            tiempos_M = []

            for i in range(n_reps):

                print(f"Graficando tiempos de ejecución M: Simulando M={M}, repeticion {i}. Método {nombre}.")

                inicio = time.perf_counter()
                metodo(M, N, S, r, sigma, B, K, x, T)
                tiempos_M.append(time.perf_counter() - inicio)

            tiempos[nombre].append(np.mean(tiempos_M))

    fig, ax = plt.subplots(figsize=(7, 5))

    for nombre, valores in tiempos.items():
        ax.plot(Ms, valores, marker="o", label=nombre)

    ax.set_xlabel("Número de simulaciones (M)")
    ax.set_ylabel("Tiempo medio (segundos)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()


    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_name)

    return fig


def graficar_tiempos_ejecucion_N(Ns, M, S, r, sigma, B, K, x, T, n_reps=1, file_name="results/convergencia_tiempos_ejecucion_N.pdf"):
    metodos = {
        "Estándar": vec_standard_monte_carlo,
        "Mejorado": vec_improved_monte_carlo
    }

    tiempos = {nombre: [] for nombre in metodos}

    for nombre, metodo in metodos.items():
        for N in Ns:
            tiempos_M = []

            for i in range(n_reps):
                print(f"Graficando tiempos de ejecución N: Simulando N={N}, repeticion {i+1}. Método {nombre}.")
                inicio = time.perf_counter()
                metodo(M, N, S, r, sigma, B, K, x, T)
                tiempos_M.append(time.perf_counter() - inicio)

            tiempos[nombre].append(np.mean(tiempos_M))

    fig, ax = plt.subplots(figsize=(7, 5))

    for nombre, valores in tiempos.items():
        ax.plot(Ns, valores, marker="o", label=nombre)

    ax.set_xlabel("Número de pasos (N)")
    ax.set_ylabel("Tiempo medio (segundos)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()

    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_name)

    return fig


def graficar_eficiencia_N(Ns, M, S, r, sigma, B, K, x, T, prima_teorica, n_reps=3, file_name="results/eficiencia_error_tiempo.pdf"):
    metodos = {
        "Estándar": vec_standard_monte_carlo,
        "Mejorado":vec_improved_monte_carlo
    }

    tiempos = {nombre: [] for nombre in metodos}
    errores = {nombre: [] for nombre in metodos}

    for nombre, metodo in metodos.items():
        for N in Ns:
            t_reps = []
            err_reps = []
            for i in range(n_reps):
                print(f"Graficando eficiencia N: Simulando N={N}, repeticion {i}. Método {nombre}.")

                inicio = time.perf_counter()
                prima = metodo(M, N, S, r, sigma, B, K, x, T)
                t_reps.append(time.perf_counter() - inicio)
                err_reps.append(np.abs(prima - prima_teorica))

            tiempos[nombre].append(np.mean(t_reps))
            errores[nombre].append(np.mean(err_reps))
    fig, ax = plt.subplots(figsize=(7,5))

    for nombre in metodos:
        ax.plot(tiempos[nombre], errores[nombre], marker=".", label=nombre)

    for nombre in metodos:
        ax.annotate(f"N={Ns[0]}", (tiempos[nombre][0], errores[nombre][0]), 
                    textcoords="offset points", xytext=(5,5), fontsize=8)

        ax.annotate(f"N={Ns[-1]}", (tiempos[nombre][-1], errores[nombre][-1]),
                    textcoords="offset points", xytext=(-25, 8), fontsize=8)

    ax.set_xlabel("Tiempo medio de ejecución (segundos)")
    ax.set_ylabel("Error absoluto medio")
    ax.set_title(f"Eficiencia computacional variando N (M={M})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()

    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_name)

    return fig
