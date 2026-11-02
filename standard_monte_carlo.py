"""
Este módulo métodos de Monte Carlo estándar para valoración de una opción digital con barrera simple cash-or-nothing. 
Específicamente, una opción put con barrera down-and-out.

Contiene dos implementaciones: 
* Monte Carlo estándar (standard_monte_carlo)
* Monte Carlo estándar vectorizado (vec_standard_monte_carlo)

Se incluye también la función theoretical_price, que calcula el precio teórico de una opción digital del mismo tipo.
"""

import numpy as np
from scipy.stats import norm
from options import CashOrNothing_DownAndOut_Put



def standard_monte_carlo(M, N, S, r, sigma, option: CashOrNothing_DownAndOut_Put):
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
        S_n = S
        for _ in range(N):
            S_n *= np.exp((r - 0.5 * sigma ** 2) * delta_t +
                          sigma * np.sqrt(delta_t) * np.random.normal())
            if option.check_barrier(S_n):
                break
        sum_payoffs += option.payoff(S_n) * np.exp(-r * option.T)
        option.reset()
    return sum_payoffs / M


def process_batch(M_batch, N, S, r, sigma, B, K, x, delta_t):
    """
    Función auxiliar de vec_standard_monte_carlo para simular un batch de trayectorias.
    """
    Z = np.random.normal(size=(M_batch, N))

    factors = np.exp((r - 0.5 * sigma ** 2) * delta_t +
                     sigma * np.sqrt(delta_t) * Z)

    paths = np.zeros((M_batch, N+1))
    paths[:, 0] = S
    paths[:, 1:] = S * np.cumprod(factors, axis=1)

    min_values = np.min(paths, axis=1)
    alive = min_values > B

    S_T = paths[:, -1]
    itm = S_T < K

    payoffs = np.where(alive & itm, x, 0)
    return payoffs


def vec_standard_monte_carlo(M, N, S, r, sigma, B, K, x, T, batch_size=10000):
    """
    Versión vectorizada de standard_monte_carlo. Considerablemente más rápido que el método anterior,
    pero el uso de memoria crece rápidamente con M y N. Se puede mitigar el crecimiento de uso de memoria 
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
        Tamaño del lote de trayectorias a simular.

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


def valor_exacto(S, r, sigma, B, K, X, T):
    """
    Calcula la prima de una opción digital cash-or-nothing down-and-out put utilizando la fórmula teórica 
    de Rubistein y Reiner (1991).
    """
    eta = 1
    phi = -1
    mu = r - sigma**2 / 2
    lam = 1 + mu / sigma**2
    v = sigma * np.sqrt(T)

    x = np.log(S / K) / v + lam * v
    x1 = np.log(S / B) / v + lam * v
    y = np.log(B**2 / (S * K)) / v + lam * v
    y1 = np.log(B / S) / v + lam * v

    D = X * np.exp(-r * T)
    c1 = D * norm.cdf(phi * x - phi * v)
    c2 = D * norm.cdf(phi * x1 - phi * v)
    c3 = D * (B / S)**(2 * lam - 2) * norm.cdf(eta * y - eta * v)
    c4 = D * (B / S)**(2 * lam - 2) * norm.cdf(eta * y1 - eta * v)

    return c1 - c2 + c3 - c4

