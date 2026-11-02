from standard_monte_carlo import standard_monte_carlo, vec_standard_monte_carlo, CashOrNothing_DownAndOut_Put, valor_exacto
from improved_monte_carlo import improved_monte_carlo

def ejemplo_1():
    """
    Estima la prima de una opción digital de tipo cash-or-nothing down-and-out put utilizando los parámetros 
    del Ejemplo 1 de Nouri et al. (2016).
    """
    option = CashOrNothing_DownAndOut_Put(B=100, K=102, x=15, T=0.5)
    price = standard_monte_carlo(
        M=100000, N=40, S=105, r=0.1, sigma=0.2, option=option)
    print(f"Prima Monte Carlo estándar: {price}")

    price_vec = vec_standard_monte_carlo(
        M=100000, N=40, S=105, r=0.1, sigma=0.2, B=100, K=102, x=15, T=0.5)
    print(f"Prima Monte Carlo estándar vectorizado: {price_vec}")

    price_improved = improved_monte_carlo(
        M=100000, N=40, S=105, r=0.1, sigma=0.2, option=CashOrNothing_DownAndOut_Put(B=100, K=102, x=15, T=0.5))
    print(f"Prima Monte Carlo mejorado: {price_improved}")

    price_theoretical = valor_exacto(
        S=105, B=100, K=102, X=15, T=0.5, r=0.1, sigma=0.2)
    print(f"Prima teórica: {price_theoretical}")


    print(f"Error relativo Monte Carlo estándar: {abs(price_vec - price_theoretical) / price_theoretical * 100:.4f}%")
    print(f"Error relativo Monte Carlo mejorado: {abs(price_improved - price_theoretical) / price_theoretical * 100:.4f}%")
