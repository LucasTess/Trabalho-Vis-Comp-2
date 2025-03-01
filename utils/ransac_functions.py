import numpy as np
import math
from .dlt_functions import compute_normalized_dlt

def RANSAC(pts1: np.ndarray, pts2: np.ndarray, dis_threshold: float, N_0: int, Ninl: float, verbose: bool = True) -> np.ndarray:
    """
    Estima uma homografia entre dois conjuntos de pontos usando o algoritmo RANSAC.

    O algoritmo itera sobre o conjunto de pontos, selecionando aleatoriamente subconjuntos de pontos, estimando a homografia usando a 
    técnica de DLT normalizada e avaliando os erros. A homografia final é determinada pelos inliers, ou seja, pontos que têm erro de 
    transformação abaixo de um limiar definido.

    Parâmetros:
    -----------
    pts1 : ndarray, shape (n_points, 2)
        Conjunto de pontos 2D da primeira imagem.
    
    pts2 : ndarray, shape (n_points, 2)
        Conjunto de pontos 2D da segunda imagem.
    
    dis_threshold : float
        Limiar de distância (erro) usado para determinar se um ponto é um inlier ou outlier.
    
    N_0 : int
        Número máximo de iterações do RANSAC.
    
    Ninl : float
        Limiar mínimo de inliers desejado para considerar a estimativa como válida.
    
    verbose : bool, opcional, padrão=True
        Se True, exibe informações sobre o andamento do algoritmo.

    Retorna:
    --------
    H : ndarray, shape (3, 3)
        Matriz de homografia estimada entre as duas imagens.
    """
    if verbose:
        print(f'Running RANSAC with {dis_threshold} tolerance...')

    success_probability = 0.99
    sample_size = 4 
    n_points = pts1.shape[0]
    best_inliers_indices: np.ndarray = np.array([])
    pts1 = pts1.squeeze()
    pts2 = pts2.squeeze()

    for iteration in range(N_0):
        indices = np.random.choice(n_points, sample_size, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]
        
        try:
            sample_H = compute_normalized_dlt(sample_pts1, sample_pts2)
        except Exception:
            continue

        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  
        pts1_transf_h = (sample_H @ pts1_h.T).T  
        pts1_transf = pts1_transf_h[:, :2] / pts1_transf_h[:, [2]]  

        errors = np.linalg.norm(pts2 - pts1_transf, axis=1)
        inliers_indices = np.where(errors < dis_threshold)[0]

        if len(inliers_indices) > len(best_inliers_indices):
            best_inliers_indices = inliers_indices
            outlier_ratio = (len(pts2) - len(inliers_indices)) / len(pts2)
            print(f"Ratio of outliers to pts2 on iteration {iteration}: {outlier_ratio}")

            required_iterations = math.log10(1 - success_probability) / math.log10(1 - (1 - outlier_ratio) ** sample_size)

            if len(inliers_indices) / len(pts2) >= Ninl:
                print("Ratio of inliers meets the Ninl threshold, stopping...")
                break
            elif iteration >= required_iterations:
                break

    H = compute_normalized_dlt(pts1[best_inliers_indices], pts2[best_inliers_indices])
    print(f"Ratio of Best inliers to pts2: {len(best_inliers_indices) / len(pts2)}")
    print(f"RANSAC ended at iteration {iteration} with {required_iterations} iterations.")
    
    return H
