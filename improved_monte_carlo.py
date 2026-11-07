"""
Este módulo contiene métodos de Monte Carlo mejorados con la técnica descripta en Nouri et al. (2016).
Estos permiten valorar opciones cash-or-nothing down-and-out put
"""

import numpy as np


def improved_monte_carlo(M, N, S, r, sigma, B, K, x, T):
    """
    Valora una opción digital cash-or-nothing down-and-out put mediante simulación de Monte Carlo estándar.
    Tiene un bajo consumo de memoria, pero es considerablemente lento.

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
    B: float
        Barrera de la opción.
    K: float
        Strike de la opción.
    x: float
        Pago de la opción.
    T: float
        Tiempo hasta la expiración.

    Retorna: float
        Prima estimada de la opción. 
    """

    delta_t = T / N
    sum_payoffs = 0

    if S <= B:
        return 0

    for _ in range(M):
        S_prev = S
        crossed_barrier = False
        for _ in range(N):
            S_next = S_prev * np.exp((r - 0.5 * sigma ** 2) * delta_t +
                                     sigma * np.sqrt(delta_t) * np.random.normal())

            if S_next <= B:
                crossed_barrier = True
                break
            else:
                p_n = np.exp(-2 * (S_prev - B) * (S_next - B) /
                             (sigma**2 * S_prev**2 * delta_t))

                if np.random.uniform() < p_n:
                    crossed_barrier = True
                    break

            S_prev = S_next

        if not crossed_barrier and S_prev < K:
            sum_payoffs += x * np.exp(-r * T)

    return sum_payoffs / M


def process_batch(M_batch, N, S, r, sigma, B, K, x, delta_t):

    Z = np.random.normal(size=(M_batch, N))

    factors = np.exp((r - 0.5 * sigma ** 2) * delta_t +
                     sigma * np.sqrt(delta_t) * Z)

    paths = np.zeros((M_batch, N+1))
    paths[:, 0] = S
    paths[:, 1:] = S * np.cumprod(factors, axis=1)

    exceeded_barrier_p = np.exp(-2 * (paths[:, :-1] - B) * (
        paths[:, 1:] - B) / (sigma**2 * paths[:, :-1]**2 * delta_t))

    mask = (paths[:, :-1] >= B) & (paths[:, 1:] >= B)

    U = np.random.uniform(size=(M_batch, N))
    crossed_bridge = np.any((U <= exceeded_barrier_p) & mask, axis=1)

    min_values = np.min(paths, axis=1)
    alive = (min_values > B) & ~crossed_bridge

    S_T = paths[:, -1]
    itm = S_T < K

    payoffs = np.where(alive & itm, x, 0)
    return payoffs


def vec_improved_monte_carlo(M, N, S, r, sigma, B, K, x, T, batch_size=10000):
    """
    Versión vectorizada de improved_monte_carlo. Considerablemente más rápida que el método anterior,
    pero el uso de memoria crece rápidamente con M y N. Se puede mitigar el crecimiento del uso de memoria 
    respecto a M eligiendo un batch_size adecuado.

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
    B: float
        Barrera de la opción.
    K: float
        Strike de la opción.
    x: float
        Pago de la opción.
    T: float
        Tiempo hasta la expiración.
    batch_size: int
        Tamaño del lote.

    Retorna: float
        Prima estimada de la opción. 
    """

    delta_t = T / N
    sum_payoffs = 0

    n_batches = M // batch_size
    remaining = M % batch_size

    for _ in range(n_batches):

        payoffs = process_batch(batch_size, N, S, r, sigma, B, K, x, delta_t)

        sum_payoffs += np.sum(payoffs)

    if remaining > 0:
        payoffs = process_batch(remaining, N, S, r, sigma, B, K, x, delta_t)
        sum_payoffs += np.sum(payoffs)

    return (sum_payoffs / M) * np.exp(-r * T)
