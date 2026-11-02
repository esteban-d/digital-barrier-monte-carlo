import numpy as np
from options import CashOrNothing_DownAndOut_Put



def improved_monte_carlo(M, N, S, r, sigma, option: CashOrNothing_DownAndOut_Put):
    """
    Valora una opción digital cash-or-nothing down-and-out put mediante simulación de Monte Carlo estándar.

    Parámetros:
    M: int
        Cantidad de trayectorias a simular.
    N: int
        Cantidad de saltos de tiempo en cada trayectoria.
    S: float
        Precio incial del subyacente.
    r: float
        Tasa libre de riesgo.
    sigma: float
        Volatilidad del subyacente.
    option: CashOrNothing_DownAndOut_Put
        Instancia de la opción a valorar.

    Retorna: float
        Prima estimada de la opción. 
    """

    delta_t = option.T / N
    sum_payoffs = 0

    option.reset()
    if option.check_barrier(S):
        return 0

    for _ in range(M):
        S_prev = S
        for _ in range(N):
            S_next = S_prev * np.exp((r - 0.5 * sigma ** 2) * delta_t +
                          sigma * np.sqrt(delta_t) * np.random.normal())

            if option.check_bridge(S_prev, S_next, delta_t, sigma):
                break

            S_prev = S_next
            
            
        sum_payoffs += option.payoff(S_next) * np.exp(-r * option.T)
        option.reset()

    return sum_payoffs / M


def vec_standard_monte_carlo(M, N, S, r, sigma, B, K, x, T, batch_size=10000):
    pass