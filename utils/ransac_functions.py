import numpy as np
import math
from .dlt_functions import compute_normalized_dlt
########################################################################################################################
# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem 
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado 
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens

########################################################################################################################



def RANSAC(pts1, pts2, dis_threshold, N_0, Ninl, verbose = True):
    if verbose:
        print(f'Running RANSAC with {dis_threshold} tolerance...')
    # Define outros parâmetros como número de amostras do modelo, probabilidades da equação de N, etc 
    p = 0.99
    s = 4
    n_points = pts1.shape[0]
    #best_H = None
    best_inliers_idx = []
    pts1 = pts1.squeeze()
    pts2 = pts2.squeeze()
    # Processo Iterativo
        # Enquanto não atende a critério de parada
        
        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2 
        
        # Usa as amostras para estimar uma homografia usando o DTL Normalizado

        # Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
        # o número de supostos inliers obtidos com o modelo estimado

        # Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas. 
        # Atualiza também o número N de iterações necessárias
    
    
    for i in range (N_0):
        #Pegando amostaras dos pontos
        indices = np.random.choice(n_points, s, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]
        
        try:
            sample_H = compute_normalized_dlt(sample_pts1,sample_pts2)
        except Exception as e:
            # Se a DLT falhar (ex.: pontos colineares), ignora essa iteração
            continue
        # Aplica homografia somente aos pts1
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # Converte para coordenadas homogêneas
        pts1_transf_h = (sample_H @ pts1_h.T).T #aplica a homografia dessa sample em pts1
        pts1_transf = pts1_transf_h[:, :2] / pts1_transf_h[:, [2]] # Normaliza para converter de volta a coordenadas 2D
        # Calcula o erro (distância Euclidiana) entre pts1 transformado e pts2
        errors = np.linalg.norm(pts2 - pts1_transf, axis=1)

        # Considera inlier se o erro for menor que o threshold
        inliers_idx = np.where(errors < dis_threshold)[0]
        #print(len(errors),len(pts2))
        if len(inliers_idx) > len(best_inliers_idx):
            best_inliers_idx = inliers_idx
            
            e = (len(pts2)-len(inliers_idx))/len(pts2) # calcula porcentagem de outliers
            print("Ratio of outliers to pts2 on i = ",i,":")
            print(e)
            N = math.log10(1-p)/math.log10(1-(1-e)**4) # recalcula o número de iterações necessárias
            
            if len(inliers_idx)/len(pts2) >= Ninl:
                print("Ratio of inliers indexes at break")
                print(len(inliers_idx)/len(pts2))
                break
            elif i >= N:
                break
    # Terminado o processo iterativo
    # Estima a homografia final H usando todos os inliers selecionados.
    print("\n #################################################### \n RANSAC RESULTS \n ####################################################")
    H = compute_normalized_dlt(pts1[best_inliers_idx], pts2[best_inliers_idx])
    print("Ratio of Best inliers to pts2:")
    print(len(best_inliers_idx)/len(pts2))
    print("Ransac ended at i = ",i,"And N = ",N)
    #return H, pts1_in, pts2_in
    return H