import numpy as np
from scipy.stats import norm

def valor_exacto(S, r, sigma, B, K, X, T):
    """
    Calcula la prima de una opción digital cash-or-nothing down-and-out put utilizando la fórmula teórica 
    de Rubistein y Reiner (1991).
    """

    if S <= B or K <= B:
        return 0.0

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
