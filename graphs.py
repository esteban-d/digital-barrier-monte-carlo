import time
import numpy as np
import matplotlib.pyplot as plt
import csv

from standard_monte_carlo import vec_standard_monte_carlo
from improved_monte_carlo import vec_improved_monte_carlo
from pathlib import Path

plt.rcParams.update({
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

DEFAULT_RESULTS_DIR = "results"
DEFAULT_RESULTS_TABLES = "tables" 

results_dir = Path(DEFAULT_RESULTS_DIR)
tables_dir = Path(DEFAULT_RESULTS_TABLES)
results_dir.mkdir(exist_ok=True)
tables_dir.mkdir(exist_ok=True)

def save_csv(x_variable_name, x, y1, y2, file_name):

    path = tables_dir/f"{file_name}.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([x_variable_name, "Estándar", "Mejorado"])
        writer.writerows(zip(x, y1, y2))

def save_csv_single_column(x_variable_name, x, y, method_name, file_name):

    path = tables_dir/f"{file_name}.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([x_variable_name, method_name])
        writer.writerows(zip(x, y))

def graficar_convergencia_N(Ns, M, S, r, sigma, B, K, x, T, prima_teorica, file_name="convergencia_N"):
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
    fig.savefig(results_dir/f"{file_name}.pdf")

    save_csv("N", Ns, primas_std, primas_mejorado, f"{file_name}_primas")
    save_csv("N", Ns, err_std, err_mejorado, f"{file_name}_err")

    return fig


def graficar_convergencia_M(Ms, N, S, r, sigma, B, K, x, T, prima_teorica, n_reps=30, file_name="convergencia_M"):

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

    media_err_std = err_std.mean(axis=1)
    media_err_mej = err_mej.mean(axis=1)
    c = media_err_mej[0] * np.sqrt(Ms[0])
    ax.plot(Ms, c/np.sqrt(Ms), "k--", label=r"Referencia $1/\sqrt{M}$")

    ax.title.set_text(f"Error (N={N})")
    ax.set_xlabel("Número de simulaciones (M)")
    ax.set_ylabel("Error absoluto medio")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/f"{file_name}.pdf")

    save_csv("M", Ms, media_err_std, media_err_mej, f"{file_name}_err")

    return fig


def graficar_tiempos_ejecucion_M(Ms, N, S, r, sigma, B, K, x, T, n_reps=10 , file_name="convergencia_tiempos_ejecucion_M"):
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
    ax.set_ylabel("Tiempo medio de ejecución (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/f"{file_name}.pdf")

    save_csv("M", Ms, tiempos["Estándar"], tiempos["Mejorado"], f"{file_name}_time")

    return fig


def graficar_tiempos_ejecucion_N(Ns, M, S, r, sigma, B, K, x, T, n_reps=1, file_name="convergencia_tiempos_ejecucion_N"):
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
    ax.set_ylabel("Tiempo medio de ejecución (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir / f"{file_name}.pdf")

    save_csv("N", Ns, tiempos["Estándar"], tiempos["Mejorado"], f"{file_name}_time")

    return fig


def graficar_deteccion_ejemplo(S, r, sigma, B, T, dots=15, N_alto=2000,
                                max_intentos=500_000, seed=None,
                                file_name="deteccion_ejemplo"):
    # busca una trayectoria donde el estándar (dots puntos) no detecte ningún cruce,
    # pero la misma simulación con N_alto sí haya cruzado la barrera en algún punto
    delta_t_N_alto = T / N_alto
    delta_t = T / dots
    idx = np.linspace(0, N_alto, dots + 1).astype(int)

    rng_master = np.random.default_rng(seed)

    for _ in range(max_intentos):
        rng = np.random.default_rng(rng_master.integers(1 << 32))
        Z = rng.normal(size=N_alto)
        log_returns = (r - 0.5 * sigma ** 2) * delta_t_N_alto + sigma * np.sqrt(delta_t_N_alto) * Z
        path_N_alto = np.concatenate(([S], S * np.exp(np.cumsum(log_returns))))
        path = path_N_alto[idx]

        if np.any(path <= B) or not np.any(path_N_alto <= B):
            continue

        seg = np.searchsorted(idx, np.argmin(path_N_alto)) - 1
        if not (0 <= seg < dots):
            continue

        S_n, S_next = path[seg], path[seg + 1]
        p_n = np.exp(-2 * (S_n - B) * (S_next - B) / (sigma ** 2 * S_n ** 2 * delta_t))
        U = rng.uniform()

        if U < p_n and p_n > 0.3:
            break
    else:
        raise RuntimeError("No se encontró una trayectoria de ejemplo adecuada.")

    t_N_alto = np.linspace(0, T, N_alto + 1)
    t = t_N_alto[idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t_N_alto, path_N_alto, color="tab:blue", linewidth=1, alpha=0.8,
            label=f"Monte Carlo sin mejorar, con $N={N_alto}$")
    ax.plot(t, path, "o-", color="tab:orange", markersize=5,
            label=f"Monte Carlo sin mejorar: puntos observados ($N={dots}$)")
    ax.plot(t[seg:seg+2], path[seg:seg+2], "o-", color="tab:red",
            linewidth=2.5, zorder=5,
            label=f"Monte Carlo mejorado ($N={dots}$): cruce detectado ($p_n={p_n:.2f}$)")
    ax.axhline(B, linestyle="--", color="gray", label="Barrera $B$")

    cruce = path_N_alto <= B
    ax.fill_between(t_N_alto, B, path_N_alto, where=cruce, color="red", alpha=0.15)

    ax.set_xlabel("Tiempo $t$")
    ax.set_ylabel("Precio del activo $S_t$")
    ax.set_title("Detección del cruce de barrera: estándar vs. mejorado")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(results_dir/f"{file_name}.pdf")

    return fig


def graficar_eficiencia_N(Ns, M, S, r, sigma, B, K, x, T, prima_teorica, n_reps=3, file_name="eficiencia_error_tiempo"):
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
        save_csv_single_column("Tiempo medio (s)", tiempos[nombre], errores[nombre], f"Error absoluto medio {nombre}", f"{file_name}_{nombre.lower()}_efficiency")
        

    for nombre in metodos:
        ax.annotate(f"N={Ns[0]}", (tiempos[nombre][0], errores[nombre][0]), 
                    textcoords="offset points", xytext=(5,5), fontsize=8)

        ax.annotate(f"N={Ns[-1]}", (tiempos[nombre][-1], errores[nombre][-1]),
                    textcoords="offset points", xytext=(-25, 8), fontsize=8)

    ax.set_xlabel("Tiempo medio de ejecución (s)")
    ax.set_ylabel("Error absoluto medio")
    ax.set_title(f"Error frente al tiempo de ejecución (M={M})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/f"{file_name}.pdf")

    return fig
