import numpy as np
from aux_functions import compute_A,normalize_points
# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
def compute_normalized_dlt(pts1, pts2):

    # Normaliza pontos
    T1, norm_points1 = normalize_points(pts1)
    T2, norm_points2 = normalize_points(pts2)
    # Constrói o sistema de equações empilhando a matrix A de cada par de pontos correspondentes normalizados
    A = compute_A(norm_points1,norm_points2)
    # Calcula o SVD da matriz A_empilhada e estima a homografia H_normalizada 
    U,S,Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    H_matrix = np.reshape(h,(3,3))
    # Denormaliza H_normalizada e obtém H
    H = np.linalg.inv(T2) @ H_matrix @ T1

    return H