import numpy as np


# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)
def normalize_points(points):
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    # Calculate the average distance of the points having the centroid as origin
    distances = np.linalg.norm(points - centroid, axis=1)
    avg_distance = np.mean(distances)
    # Define the scale to have the average distance as sqrt(2)
    scale = np.sqrt(2) / avg_distance
    # Define the normalization matrix (similar transformation)
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    # Normalize points
    norm_pts = (T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :2]
    return T, norm_pts

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def compute_A(pts1, pts2):
    # Add homogeneous coordinates
    pts1 =np.hstack((pts1,np.ones((pts1.shape[0],1)))).T
    pts2 =np.hstack((pts2,np.ones((pts2.shape[0],1)))).T
    # Compute matrix A
    npoints = pts1.shape[1]
    A = np.zeros((3*npoints,9))
    for k in range(npoints):
      A[3*k,3:6] = -pts2[2,k]*pts1[:,k]
      A[3*k,6:9] = pts2[1,k]*pts1[:,k]
      A[3*k+1,0:3] = pts2[2,k]*pts1[:,k]
      A[3*k+1,6:9] = -pts2[0,k]*pts1[:,k]
      A[3*k+2,0:3] = -pts2[1,k]*pts1[:,k]
      A[3*k+2,3:6] = pts2[0,k]*pts1[:,k]   
    return A