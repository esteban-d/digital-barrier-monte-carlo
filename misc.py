"""
Este módulo contiene funciones auxiliares para calcular la pendiente de los tiempos de ejecución de los métodos de Monte Carlo estándar y mejorado, 
de la recta que mejor se ajusta a las mediciones por cuadrados mínimos. 
"""


import numpy as np


def calcular_pendiente_cuadrados_minimos(file):
    N, estandar, mejorado = np.loadtxt("tables/" + file, delimiter=",", skiprows=1, unpack=True)

    pendientes = {}
    for metodo, tiempos in zip(["Estandar", "Mejorado"], [estandar, mejorado]):
        pendiente, _ = np.polyfit(N, tiempos, 1)
        print(f"Pendiente de {metodo}: {pendiente}")
        pendientes[metodo] = pendiente
    print(f"El cociente de sobrecarga (pendiente mejorado / pendiente estandar) es: {pendientes['Mejorado'] / pendientes['Estandar']}")
    


def calcular_pendientes():
    files = ["tiempos_ejecucion_N_time.csv", "tiempos_ejecucion_M_time.csv"]
    for file in files:
        print(f"Calculando pendiente para {file}:")
        calcular_pendiente_cuadrados_minimos(file)
        print()