import numpy as np
import argparse
import glob
import cv2 as cv
import os
import sys
from PIL import Image, ImageFilter
import imutils

#Generator for later to create names sequentially in certain format
def file_name_generator():
    x = 1
    while x > 0:
        yield "sample"+str(x)
        x += 1

#Uses Canny edge detection and lines up edges of each picture with the previous
#
# For each image, Canny edge detection is used to find the blade edge
# This is done with automatically found values ( sigma, lower, upper ) and
# using OpenCV's Canny function. 
#
# Because of edges detected on the surface of the blade, we iterate from through the
# image's pixels from the bottom up to find the first pixel where the value does not
# equate to 0 (indicating a black pixel).
#
# For the first image, we crop the image so that the first pixel of the blade's edge is centered
# given a certain "window" height. Try to keep "window" as an even number for precision.
#
# Example: Given window height 500 and a perfectly straight edge, each pixel of the edge
# would be positioned at a y value of 250 in the cropped image.
#
# For each subsequent image, the image is cropped according to the previous image's
# final blade location to align the blade through each of the pictures. In other words,
# if the blade edge in the first picture is ends at pixel height 400, the blade edge of the
# next image will begin at 400.
#
# Calculated using previous pixel size and window size
#
# Images saved in directory as file_name_edges_cropped.tif
def detect_edges_and_crop():
    first = True
    window = 400
    for x in ['sample3.tif', 'sample4.tif', 'sample5.tif', 'sample6.tif']:
        img = cv.imread(x)
        sigma = .33
        v = np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv.Canny(img, lower, upper)
        height = np.size(edges, 0)
        width= np.size(edges, 1)
        for y in reversed(range(height)):
            if edges[y][0] > 0:
                center_location = y
                break
        if first:
            first = False
            edges = edges[center_location - int((window/2)):center_location + int((window/2)), 0:width]
            height = np.size(edges, 0)
            width= np.size(edges, 1)
            for y in reversed(range(height)):
                if edges[y][width-1] > 0:
                    prev_edge_pixel = y
                    break
        else:
            edges = edges[center_location - prev_edge_pixel:center_location + (window-prev_edge_pixel), 0:width]
            height = np.size(edges, 0)
            width= np.size(edges, 1)
            for y in reversed(range(height)):
                if edges[y][width-1] > 0:
                    prev_edge_pixel = y
                    break
        cv.imwrite(x[0:-4]+"_edges_cropped.tif", edges)       

# Stitches edge-detected and cropped images together
#
# Uses Pillow (PIL) to stitch images together.
# Implementation is creating an image of size (image height x sum of umage widths)
# and pasting the pictures in one by one.
def stitch():
    # 
    images = list(map(Image.open, ['sample3_edges_cropped.tif', 'sample4_edges_cropped.tif', 'sample5_edges_cropped.tif', 'sample6_edges_cropped.tif']))    
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0

    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save('test.tif')

# Finds damage overall in final test given a threshold
#
# Given an image of a blade, function (ideally) finds location
# of ideal pixel location (Similar to how detect_edges_and_crop()
# finds an edge.
#
# Implementation can be improved with use of slope from the undamaged start of a blade
# and the undamaged end of blade. (Avoids problem of detecting damaged area as ideal pixel
# location.
#
# Finds damage linearly.
#
# For each pixel along the edge's horizontal axis, the program scans the image vertically
# for pixels. This scan's size is determined by the threshold. If a pixel indicating an
# edge is found within the threshold, that vertical segment of the blade is considered
# undamaged. These values are summed and used to calculate the percent damage.
def find_damage():

    img = cv.imread('test.tif')
    cv.imshow('image', img)

    sigma = .33

    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv.Canny(img, lower, upper)

    cv.imwrite('sample_edges.tif', edges)
    height = np.size(edges, 0)
    width= np.size(edges, 1)

    threshold = 10

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

    print(str(int(((found_false / (found_true + found_false))*10000)//100)) + '%')

    cv.imshow("test", np.hstack([edges]))


