# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: 
# Lucas da Rosa Tessari
# João Gabriel Santos Custódio


# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv
from utils.dlt_functions import compute_normalized_dlt
from utils.ransac_functions import RANSAC
# Exemplo de Teste da função de homografia usando o SIFT
MIN_MATCH_COUNT = 10
img1 = cv.imread('imgs/livro_001.jpg', 0)   # queryImage
img2 = cv.imread('imgs/livro_002.jpg', 0)        # trainImage
# img1 = cv.imread('imgs/comicsStarWars01.jpg', 0)   # queryImage
# img2 = cv.imread('imgs/comicsStarWars02.jpg', 0)        # trainImage

Ninl = 0.90 # Razão limite entre inliers e total de pontos
N_0 = 1000 # Esse é o N inicial, o máximo para condição de parada
dis_threshold = 3

# Inicialização do SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)



if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
    
    #################################################
    M = RANSAC(src_pts,dst_pts,dis_threshold,N_0,Ninl) # AQUI ENTRA A SUA FUNÇÃO DE HOMOGRAFIA!!!!
    #################################################

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################
