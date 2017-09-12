# -*- coding: utf-8 -*-
"""
Created on Wed May 03 13:50:55 2017

@author: Manasa Kadiri
"""

import numpy as np
from math import sqrt, floor
from PIL import Image, ImageFilter
from scipy import ndimage
from random import shuffle
import cairo
import logging
from os import path
from time import time
import logging
import argparse
import painterly 
import matplotlib.pyplot as plt

class Stroke:

    def __init__(self, coord):
        self.points = [coord]
        self.color = None

    def cairo_convert(self, width, height):
        
        cairo_points = []
    
        for point in self.points:
            cairo_points.append((point[0]/width,point[1]/height))
            
        self.points = cairo_points



def image_diff(img_a, img_b):
    
    width, height = img_a.size
    
    # convert images to numpy arrays
    array_a = np.asarray(img_a)
    array_b = np.asarray(img_b)
    
    # calculate distance
    array_diff = array_a - array_b
    array_diff = array_diff**2
    array_diff = array_diff.reshape(height, width, 3)
    array_diff = np.sum(array_diff, axis=2)
    array_diff = np.sqrt(array_diff)
    
    return array_diff
    
THRESHOLD = 10
GRID_SIZE = 1
MIN_STROKE_LEN = 4
MAX_STROKE_LEN = 16
q_radii = {
    'low': [64, 32, 16, 8],
    'medium': [64, 16, 8, 4],
    'high': [64, 8, 4, 2]}

radi = q_radii['medium']
in_filename='./test-images/haruhi.png'

newPainting = painterly.Painting(in_filename)
radi=64
newPainting.reference_img = newPainting.source_img.filter(ImageFilter.GaussianBlur(radius=radi))
#newPainting.reference_img.show()
ww = newPainting.surface.get_width()
hh = newPainting.surface.get_height()
strokes = []
newPainting.difference_img = image_diff(newPainting.reference_img, newPainting.canvas)
#plt.imshow(newPainting.difference_img)
#plt.show()
grid = GRID_SIZE * radi
x_grid = np.arange(0, (ww//grid)*grid, grid)
y_grid = np.arange(0, (hh//grid)*grid, grid)
for x in x_grid:
    for y in y_grid:
        section = newPainting.difference_img[y:y+grid, x:x+grid]
        total_error = section.sum() // (grid**2)
                                
        if (total_error > THRESHOLD or radi == 64):
            section = section.reshape((grid, grid))
            max_coord = np.argmax(section)
            max_coord = (x+(max_coord%grid), y+(max_coord//grid))
#                    print max_coord
            stroke = Stroke(max_coord)
            strokes.append(stroke)



