import modulo as mod
import numpy as np
np.random.seed(42)
N = 64
n = 500
limiar = mod.encontrar_limiar(N, n)
print(f'Limiar finito para {N}: {limiar}')

# Obs.: o valor de p foi escolhido como sendo a estimativa do limiar finito de uma matriz de lado 64

simulacao = mod.simulacao_percolacao(N, limiar)
print(f'Percolou: {simulacao[0]} | Tamanho do maior cluster: {simulacao[1]} | Total de clusters: {simulacao[2]}')

mod.grafico_tam_clusters(N, limiar, n)  # gráficos sobre o maior cluster obtido em cada matriz
mod.grafico_num_clusters(N, limiar, n)  # gráficos sobre o número total de clusters obtido em cada matriz

