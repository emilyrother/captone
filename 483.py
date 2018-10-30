import numpy as np
import argparse
import glob
import cv2 as cv
from matplotlib import pyplot as plt
import os

img = cv.imread('sample3.tif',0)

sigma = .33
# compute the median of the single channel pixel intensities
v = np.median(img)
 
# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv.Canny(img, lower, upper)

#edges = cv.Canny(img,2,100)

cv.imwrite('sampleedges.jpg', edges)
height = np.size(edges, 0)
width= np.size(edges, 1)

threshold = 5

for x in reversed(range(height)):   # tries to detect line
    if edges[x][150] > 0:           # can be changed to find averages
        edge_height = x             # or average on a line
        break                       # or compare with unused blade
    
found_true = 0
found_false = 0

for x in range(width):  
    found = False
    for y in range(edge_height - threshold, edge_height + threshold):
        if edges[y][x] > 1:
            found = True
            break
    if found:
        found_true += 1
    else:
        found_false += 1

print(found_false / (found_true + found_false))

cv.imshow("test", np.hstack([edges]))

# import the necessary packages

