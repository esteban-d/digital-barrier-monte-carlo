from standard_monte_carlo import standard_monte_carlo, vec_standard_monte_carlo, valor_exacto
from improved_monte_carlo import improved_monte_carlo, vec_improved_monte_carlo
import numpy as np

def ejemplo_1():
    """
    Estima la prima de una opción digital de tipo cash-or-nothing down-and-out put utilizando los parámetros 
    del Ejemplo 1 de Nouri et al. (2016).
    """

    np.random.seed(40) 

    M = 100000  # Número de trayectorias
    N = 50      # Número de pasos de tiempo
    S = 105     # Precio inicial del subyacente
    r = 0.1     # Tasa libre de riesgo
    sigma = 0.2 # Volatilidad del subyacente
    T = 0.5     # Tiempo hasta el vencimiento
    B = 100
    K = 102
    x = 15
    


    #price = standard_monte_carlo(
    #    M=M, N=N, S=S, r=r, sigma=sigma,  B=B, K=K, x=x, T=T)
    #print(f"Prima Monte Carlo estándar: {price}")

    price_vec = vec_standard_monte_carlo(
        M=M, N=N, S=S, r=r, sigma=sigma, B=B, K=K, x=x, T=T)
    print(f"Prima Monte Carlo estándar vectorizado: {price_vec}")

    #price_improved = improved_monte_carlo(
    #    M=M, N=N, S=S, r=r, sigma=sigma, B=B, K=K, x=x, T=T)
    #print(f"Prima Monte Carlo mejorado: {price_improved}")
    
    price_improved = vec_improved_monte_carlo(
        M=M, N=N, S=S, r=r, sigma=sigma, B=B, K=K, x=x, T=T)
    print(f"Prima Monte Carlo mejorado vectorizado: {price_improved}")

    price_theoretical = valor_exacto(
        S=S, B=B, K=K, X=x, T=T, r=r, sigma=sigma)
    print(f"Prima teórica: {price_theoretical}")

    print(f"Error relativo Monte Carlo estándar: {abs(price_vec - price_theoretical) / price_theoretical * 100:.4f}%")
    print(f"Error relativo Monte Carlo mejorado: {abs(price_improved - price_theoretical) / price_theoretical * 100:.4f}%")

