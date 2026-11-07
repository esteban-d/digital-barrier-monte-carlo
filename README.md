# Monte Carlo modificado para opciones digitales con barrera

Trabajo de **Modelos Matemáticos en Finanzas Cuantitativas**, FaMAF, UNC. Se estudia e implementa la valoración de una opción 
cash-or-nothing down-and-out put  utilizando el algoritmo de Monte Carlo modificado de Nouri et al., comparando su velocidad 
de convergencia y error frente a la simulación de Monte Carlo estándar y la fórmula analítica exacta de Rubinstein-Reiner como validación.

## Ejecución

Si bien se recomienda consultar directamente las carpetas results y tables para obtener los gráficos y las mediciones directamente, es posible
reproducir estos resultados ejecutando el siguiente comando desde la raíz del repositorio. 

```bash
python experiments.py
```

Los tiempos de ejecución pueden variar según el hardware, el sistema operativo y la carga del sistema.

## Autores

- Facundo Otamendi
- Esteban Druetta

## Referencias

- Nouri, K.; Abbasi, B.; Omidi, F.; Torkzadeh, L. (2016). [Digital barrier options pricing: an improved Monte Carlo algorithm](https://link.springer.com/article/10.1007/s40096-016-0179-8). *Mathematical Sciences*, 10(2), 65–70.
- Rubinstein, M.; Reiner, E. (1991). Unscrambling the binary code. *Risk Magazine*, 4, 75–83.
