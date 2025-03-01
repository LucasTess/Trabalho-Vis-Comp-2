import numpy as np
from typing import Tuple


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normaliza um conjunto de pontos 2D em coordenadas homogêneas.

    A normalização é feita de forma que o centro de massa seja a origem e a média das distâncias
    entre os pontos e a origem seja igual a sqrt(2).

    Parâmetros:
    -----------
    points : ndarray, shape (n_points, 2)
        Conjunto de pontos 2D a serem normalizados.
    
    Retorna:
    --------
    T : ndarray, shape (3, 3)
        Matriz de transformação usada para normalizar os pontos.
    
    norm_pts : ndarray, shape (n_points, 2)
        Conjunto de pontos normalizados.
    """
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    avg_distance = np.mean(distances)
    scale = np.sqrt(2) / avg_distance

    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    norm_pts = (T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :2]
    
    return T, norm_pts


def compute_A(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Monta a matriz A do sistema de equações do DLT.

    A matriz A é usada na técnica de DLT (Direct Linear Transformation) para calcular a homografia
    entre dois conjuntos de pontos correspondentes.

    Parâmetros:
    -----------
    pts1 : ndarray, shape (n_points, 2)
        Conjunto de pontos 2D da primeira imagem.
    
    pts2 : ndarray, shape (n_points, 2)
        Conjunto de pontos 2D da segunda imagem.
    
    Retorna:
    --------
    A : ndarray, shape (3 * n_points, 9)
        Matriz do sistema de equações para estimar a homografia.
    """
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T
    npoints = pts1.shape[1]
    A = np.zeros((3 * npoints, 9))

    for k in range(npoints):
        A[3 * k, 3:6] = -pts2[2, k] * pts1[:, k]
        A[3 * k, 6:9] = pts2[1, k] * pts1[:, k]
        A[3 * k + 1, 0:3] = pts2[2, k] * pts1[:, k]
        A[3 * k + 1, 6:9] = -pts2[0, k] * pts1[:, k]
        A[3 * k + 2, 0:3] = -pts2[1, k] * pts1[:, k]
        A[3 * k + 2, 3:6] = pts2[0, k] * pts1[:, k]

    return A
