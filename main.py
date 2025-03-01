# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome:
# Lucas da Rosa Tessari
# João Gabriel Santos Custódio

import numpy as np
import cv2 as cv
from utils.plot_images import plot_images
from utils.ransac_functions import RANSAC
from utils.parser import Parser


MIN_MATCH_COUNT = 10
COLOR = (0,255,0)

parser = Parser().get_file()

print(parser.train_image, parser.query_image)

img1 = cv.imread(parser.train_image, 0)
img2 = cv.imread(parser.query_image, 0)

INLIERS_TOTAL_POINTS_RATIO = 0.90
INITIAL_NUMBERS_OF_INTERACTIONS = 1000
DISTANCE_LIMIT_THRESHOLD = 3

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
    
    M = RANSAC(src_pts,dst_pts, DISTANCE_LIMIT_THRESHOLD, INITIAL_NUMBERS_OF_INTERACTIONS, INLIERS_TOTAL_POINTS_RATIO)

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

 # draw matches in green color
draw_match_params = dict(matchColor = COLOR,
                   singlePointColor = None,
                   flags = 2)

img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_match_params)

plot_images(img1, img2, img3, img4)
