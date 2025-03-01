import numpy as np
from .aux_functions import compute_A, normalize_points

def compute_normalized_dlt(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Estima a homografia entre dois conjuntos de pontos usando o método DLT normalizado.

    A função normaliza os pontos de entrada, calcula a matriz A a partir dos pontos normalizados, 
    realiza a decomposição SVD para estimar a homografia e, finalmente, denormaliza a homografia.

    Parâmetros:
    -----------
    points1 : ndarray, shape (n_points, 2)
        Conjunto de pontos 2D da primeira imagem.
    
    points2 : ndarray, shape (n_points, 2)
        Conjunto de pontos 2D da segunda imagem.

    Retorna:
    --------
    H : ndarray, shape (3, 3)
        Matriz de homografia estimada entre as duas imagens.
    """
    normalization_matrix1, normalized_points1 = normalize_points(points1)
    normalization_matrix2, normalized_points2 = normalize_points(points2)
    
    matrix_A = compute_A(normalized_points1, normalized_points2)
    
    _, _, Vt = np.linalg.svd(matrix_A)
    homography_vector = Vt[-1, :]
    
    normalized_homography_matrix = np.reshape(homography_vector, (3, 3))
    
    homography_matrix = np.linalg.inv(normalization_matrix2) @ normalized_homography_matrix @ normalization_matrix1

    return homography_matrix
