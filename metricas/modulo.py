import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


def simulacao_percolacao(N, p):
  ''' Recebe os parâmetros N (lado da grade da matriz) e p (probabilidade da entrada da matriz ser 1, e 0 caso contrário).
      Retorna se a matriz percolou (True/False), o número de clusters e o tamanho do maior cluster'''
  percolou = False
  maior_cluster = 0
  num_clusters = 0

  grade = np.random.rand(N, N) < p
  grade = np.where(grade, 1, 0)
  if not np.any(grade):
    return False, 0, 0
  # Se a matriz for inteiramente de zeros, já retorna que não percola e não teve clusters

  estrutura = ndimage.generate_binary_structure(rank=2, connectivity=1)
  # Define a estrutura de Vizinhos de von Neumann

  grade_clusters, num_clusters = ndimage.label(grade, estrutura)
  # Função de etiquetar as matrizes em clusters
  coluna_1 = grade_clusters[:, 0]
  coluna_n = grade_clusters[:, -1]
  linha_1 = grade_clusters[0]
  linha_n = grade_clusters[-1]

  intersecoes_vertical = np.intersect1d(linha_1, linha_n)
  intersecoes_horizontal = np.intersect1d(coluna_1, coluna_n)

  percolou_vertical = np.any(intersecoes_vertical > 0)
  percolou_horizontal = np.any(intersecoes_horizontal > 0)
  if percolou_vertical or percolou_horizontal:
    percolou = True

  grade_clusters_d1 = np.ravel(grade_clusters)

  contagem = np.bincount(grade_clusters_d1)
  # Conta ocupação de cada etiqueta de cluster
  maior_cluster = max(contagem[1:])
  # Maior cluster é o elemento que ocupou mais sítios (excluindo o zero)

  return percolou, maior_cluster, num_clusters


def metricas_cluster(N, p, n):
  ''' Recebe os parâmetros N (lado da grade da matriz), p (probabilidade da entrada da matriz ser 1, e 0 caso contrário), e n (número de amostras a serem geradas). 
      Retorna métricas sobre as matrizes que percolam e que não percolam, separadamente: o tamanho dos maiores clusters, a probabilidade de percolar/não percolar, a média do tamanho dos maiores clusters, total de matrizes que percolam ou não e média do total de clusters'''
  tam_cluster_percola = []
  tam_cluster_nao_percola = []
  num_cluster_percola = []
  num_cluster_nao_percola = []
  total_percola, total_nao_percola = 0, 0
    
  for i in range(n):
    y = simulacao_percolacao(N, p)
    if y[0] == True:
      tam_cluster_percola.append(y[1])
      num_cluster_percola.append(y[2])
      total_percola += 1
    else:
      tam_cluster_nao_percola.append(y[1])
      num_cluster_nao_percola.append(y[2])
      total_nao_percola += 1

  prob_percolar = total_percola / n
  prob_nao_percolar = 1 - prob_percolar

  media_tam_percola = sum(tam_cluster_percola) / total_percola if total_percola else 0
  media_tam_nao_percola = sum(tam_cluster_nao_percola) / total_nao_percola if total_nao_percola else 0

  media_num_percola = sum(num_cluster_percola) / total_percola if total_percola else 0
  media_num_nao_percola = sum(num_cluster_nao_percola) / total_nao_percola if total_nao_percola else 0
    
  return tam_cluster_percola, tam_cluster_nao_percola, prob_percolar, prob_nao_percolar, media_tam_percola, media_tam_nao_percola, total_percola, total_nao_percola, num_cluster_percola, num_cluster_nao_percola, media_num_percola, media_num_nao_percola


def grafico_tam_clusters(N, p, n):
  ''' Recebe os parâmetros N (lado da grade da matriz), p (probabilidade da entrada da matriz ser 1, e 0 caso contrário), e n (número de amostras a serem geradas).
      Retorna gráficos contendo métricas e a distribuição dos maiores clusters para os casos em que percola (esquerda) e não percola (direita)'''
  dados = metricas_cluster(N, p, n)
  tam_cluster_percola, tam_cluster_nao_percola, prob_percolar, prob_nao_percolar, media_tam_percola, media_tam_nao_percola, total_percola, total_nao_percola, num_cluster_percola, num_cluster_nao_percola, media_num_percola, media_num_nao_percola = dados

  fig, axes = plt.subplots(1, 2, figsize=(14, 5))

  # Histograma percolação
  axes[0].hist(tam_cluster_percola, bins=30, edgecolor='black')
  axes[0].set_title(f"Percolou\nP={prob_percolar:.3f} | Média={media_tam_percola:.2f} | Total de matrizes: {total_percola}")
  axes[0].set_xlabel("Tamanho do maior cluster")
  axes[0].set_ylabel("Frequência")
  axes[0].grid(True)

  # Histograma NÃO percolação
  axes[1].hist(tam_cluster_nao_percola, bins=30, edgecolor='black')
  axes[1].set_title(f"Não Percolou\nP={prob_nao_percolar:.3f} | Média={media_tam_nao_percola:.2f} | Total de matrizes: {total_nao_percola}")
  axes[1].set_xlabel("Tamanho do maior cluster")
  axes[1].grid(True)

  plt.suptitle(f"Distribuição do tamanho dos maiores clusters (N={N}, p={p}, n={n})", fontsize=16)
  plt.tight_layout()
  plt.show()


def grafico_num_clusters(N, p, n):
  ''' Recebe os parâmetros N (lado da grade da matriz), p (probabilidade da entrada da matriz ser 1, e 0 caso contrário), e n (número de amostras a serem geradas).
      Retorna gráficos contendo métricas e a distribuição do número total de clusters para os casos em que percola (esquerda) e não percola (direita)'''
  dados = metricas_cluster(N, p, n)
  tam_cluster_percola, tam_cluster_nao_percola, prob_percolar, prob_nao_percolar, media_tam_percola, media_tam_nao_percola, total_percola, total_nao_percola, num_cluster_percola, num_cluster_nao_percola, media_num_percola, media_num_nao_percola = dados

  fig, axes = plt.subplots(1, 2, figsize=(14, 5))

  # Histograma percolação
  axes[0].hist(num_cluster_percola, bins=30, edgecolor='black')
  axes[0].set_title(f"Percolou \nMédia={media_num_percola:.2f} | Total de matrizes: {total_percola}")
  axes[0].set_xlabel("Total de clusters")
  axes[0].set_ylabel("Frequência")
  axes[0].grid(True)

  # Histograma NÃO percolação
  axes[1].hist(num_cluster_nao_percola, bins=30, edgecolor='black')
  axes[1].set_title(f"Não percolou \nMédia={media_num_nao_percola:.2f} | Total de matrizes: {total_nao_percola}")
  axes[1].set_xlabel("Número total de clusters")
  axes[1].grid(True)

  plt.suptitle(f"Distribuição do número de clusters (N={N}, p={p}, n={n})", fontsize=16)
  plt.tight_layout()
  plt.show()


def encontrar_limiar(N, n):
  ''' Recebe os parâmetros N (lado da grade da matriz) e n (número de amostras a serem geradas).
      Retorna o limiar finito de N (x)'''
  erro = 0.01
  def f(N, x, n):
    return metricas_cluster(N, x, n)[2]  # retorna probabilidade de percolar

  a, b = 0, 1
  x = (a + b)/2

  while abs(f(N, x, n)-0.5) > erro:
    resultado = f(N, x, n)
    if resultado < 0.5:
      a = x
    elif resultado > 0.5:
      b = x
    x = (a + b)/2
  return x

def limiar_critico(n):
  ''' Recebe como parâmetro n (número de amostras a serem geradas).
      Renorna a estimativa do limiar crítico, quando N tende ao infinito'''
  lista_N = [16, 64, 128, 256, 512, 1024]
  lista_limiares = []

  for N in lista_N:
    x = encontrar_limiar(N, n)
    lista_limiares.append(x)
    print(lista_limiares)

  plt.figure(figsize=(8,8))
  plt.xlabel('N')
  plt.ylabel('Limiar pc(N)')
  plt.title(f'Limiares finitos versus N')
  plt.plot(lista_N, lista_limiares, marker='o')
  for x, y in zip(lista_N, lista_limiares):
    plt.text(x, y, f"({x}, {y:.5f})", ha='left', va='bottom')
  plt.grid(True)
  plt.show()



