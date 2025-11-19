import modulo as mod
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

# Estabelecimento de parâmetros
N = 512
p = 0.6
n = 500

# Boxplot para o tamanho do maior cluster obtido em cada simulação
dados1 = []
for i in range(n):
    dados1.append(mod.simulacao_percolacao(N, p)[1])

dados1 = np.array(dados1)

plt.boxplot(dados1, patch_artist=True, boxprops=dict(facecolor='Orange'), medianprops=dict(color='black', linewidth=2))
plt.title(f'Boxplot do tamanho do maior cluster em cada uma das \n{n} simulações com N = {N} e p = {p}')
plt.grid(True)
plt.show()

print(f"Medidas resumo do tamanho do maior cluster em cada uma das {n} simulações com N = {N} e p = {p}")
print(f'Mínimo: {dados1.min()} | 1º Q: {np.quantile(dados1, 0.25)} | Mediana: {np.quantile(dados1, 0.5)} | 3º Q: {np.quantile(dados1, 0.75)} | Máximo: {dados1.max()}')

# Boxplot para o total de clusters obtido em cada simulação
dados2 = []
for i in range(n):
    dados2.append(mod.simulacao_percolacao(N, p)[2])

dados2 = np.array(dados2)

plt.boxplot(dados2, patch_artist=True, boxprops=dict(facecolor='Orange'), medianprops=dict(color='black', linewidth=2))
plt.title(f'Boxplot do total de clusters em cada uma das \n{n} simulações com N = {N} e p = {p}')
plt.grid(True)
plt.show()

print(f"Medidas resumo do total de clusters em cada uma das {n} simulações com N = {N} e p = {p}")
print(f'Mínimo: {dados2.min()} | 1º Q: {np.quantile(dados2, 0.25)} | Mediana: {np.quantile(dados2, 0.5)} | 3º Q: {np.quantile(dados2, 0.75)} | Máximo: {dados2.max()}')
