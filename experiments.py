"""
Módulo principal. Permite ejecutar los experimentos para el ejemplo 1 de Nouri et al. (2016) y graficar los resultados.
"""

from standard_monte_carlo import vec_standard_monte_carlo
from improved_monte_carlo import vec_improved_monte_carlo
from graphs import graficar_convergencia_M, graficar_convergencia_N, graficar_tiempos_ejecucion_M, graficar_eficiencia_N, graficar_tiempos_ejecucion_N, graficar_deteccion_ejemplo
from exact_value import valor_exacto

import numpy as np


DEF_SEED = 123

# Parámetros del la opción elegida
M = 1000000  # Número de simulaciones
N = 50      # Número de pasos de tiempo
S = 105     # Precio inicial del subyacente
r = 0.1     # Tasa libre de riesgo
sigma = 0.2  # Volatilidad del subyacente
T = 0.5     # Tiempo hasta el vencimiento
B = 100 # Barrera
K = 102 # Strike
x = 15  # Pago

# Valor teórico
V = valor_exacto(S=S, B=B, K=K, X=x, T=T, r=r, sigma=sigma)


def calcular_prima_ejemplo():
    """
    Estima la prima de una opción digital de tipo cash-or-nothing down-and-out put con los parámetros 
    del Ejemplo 1 de Nouri et al. (2016).
    """
    
    price_vec = vec_standard_monte_carlo(
        M=M, N=N, S=S, r=r, sigma=sigma, B=B, K=K, x=x, T=T)
    print(f"Prima Monte Carlo estándar vectorizado: {price_vec}")

    price_improved = vec_improved_monte_carlo(
        M=M, N=N, S=S, r=r, sigma=sigma, B=B, K=K, x=x, T=T)
    print(f"Prima Monte Carlo mejorado vectorizado: {price_improved}")

    price_theoretical = V
    print(f"Prima teórica: {price_theoretical}")

    print(
        f"Error relativo Monte Carlo estándar: {abs(price_vec - price_theoretical) / price_theoretical * 100:.4f}%")
    print(
        f"Error relativo Monte Carlo mejorado: {abs(price_improved - price_theoretical) / price_theoretical * 100:.4f}%")


def graficar_convergecia_N_ejemplo():
    """
    Grafica la velocidad de convergencia de los métodos de Monte Carlo estándar y mejorado con respecto al número de pasos temporales N.
    """

    Ns = np.unique(np.geomspace(5, 1000, 20).astype(int))
    graficar_convergencia_N(Ns, M, S, r, sigma, B, K, x, T, V)
    print("Gráfico convergecia_N generado")

def graficar_convergecia_M_ejemplo():
    """
    Grafica la velocidad de convergencia de los métodos de Monte Carlo estándar y mejorado con respecto al número de simulaciones M.
    """
    Ms = np.unique(np.geomspace(1_000, 1_000_000, 10).astype(int))
    graficar_convergencia_M(Ms, N, S, r, sigma, B, K, x, T, V)
    print("Gráfico convergencia_M generado")
        

def graficar_tiempos_ejecucion_M_ejemplo():
    """
    Grafica los tiempos de ejecución de los métodos de Monte Carlo estándar y mejorado con respecto al número de simulaciones M.
    """
    Ms = np.arange(1_000, 1_000_001, 100_000)
    graficar_tiempos_ejecucion_M(Ms, N, S, r, sigma, B, K, x, T)
    print("Gráfico tiempos_de_ejecucion_M generado")

def graficar_tiempos_ejecucion_N_ejemplo(): 
    """
    Grafica los tiempos de ejecución de los métodos de Monte Carlo estándar y mejorado con respecto al número de pasos temporales N.
    """
    Ns = np.arange(5, 1001, 50)
    graficar_tiempos_ejecucion_N(Ns, M, S, r, sigma, B, K, x, T)
    print("Gráfico tiempos_de_ejecucion_N generado")


def graficar_eficiencia_ejemplo():
    """
    Grafica el error relativo de los métodos con respecto al tiempo de ejecución, para distintos valores de N.
    """
    Ns = np.unique(np.geomspace(5, 1000, 8).astype(int))
    graficar_eficiencia_N(Ns, M, S, r, sigma, B, K, x, T, V)
    print("Gráfico eficiencia_N generado")


def graficar_deteccion_ejemplo_ejemplo():
    """
    Grafica un ejemplo didáctico que ilustra el funcionamiento del método mejorado y el sesgo que se introduce al no detectar
    cruces de barrera entre nodos discretos consecutivos en el método estándar.
    """
    graficar_deteccion_ejemplo(S, r, sigma, B, T, dots=15, seed=22)
    print("Gráfico deteccion_ejemplo generado")


if __name__ == "__main__":
    np.random.seed(DEF_SEED)

    graficar_convergecia_N_ejemplo()
    graficar_convergecia_M_ejemplo()
    graficar_tiempos_ejecucion_M_ejemplo()
    graficar_tiempos_ejecucion_N_ejemplo()
    graficar_eficiencia_ejemplo()
    graficar_deteccion_ejemplo_ejemplo()

    calcular_prima_ejemplo()
