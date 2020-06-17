import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage import transform as tf
    
def pad(img, crd, resize_val): # pad image with constant border and calculates new bounding box coordinates
    
    pad_y, pad_x = int(img.shape[0]/resize_val), int(img.shape[1]/resize_val)
    img_padded = cv2.copyMakeBorder(img,pad_y,pad_y,pad_x,pad_x,cv2.BORDER_CONSTANT)
    img_padded = cv2.resize(img_padded, (img.shape[1], img.shape[0]))
    
    crd[0],crd[1],crd[2],crd[3] = crd[0]+pad_x,crd[1]+pad_y,crd[2]+pad_x,crd[3]+pad_y
    new_crd = (resize_val/(resize_val+2)) * crd
    
    return img_padded, new_crd

def shear(img, crd, shear_val, dir_xy): # shear image and calculates new bounding box coordinates
    
    if dir_xy == 0:
        arr = np.array([[1,shear_val,0], [0,1,0], [0,0,1]])
    else:
        arr = np.array([[1,0,0], [shear_val,1,0], [0,0,1]])
        
    afine_tf = tf.AffineTransform(matrix = arr)
    img_sheared = tf.warp(img, inverse_map=afine_tf)
    img_sheared = img_as_ubyte(img_sheared)
    
    if dir_xy == 0:
        crd[0] = crd[0] - shear_val * crd[1]
        crd[2] = crd[2] - shear_val * crd[3]
        if crd[0] < 0:
            crd[0] = 0
        if crd[2] >= img.shape[1]:
            crd[2] = img.shape[1]-1
    else:
        crd[1] = crd[1] - shear_val * crd[0]
        crd[3] = crd[3] - shear_val * crd[2]
        if crd[1] < 0:
            crd[1] = 0
        if crd[3] >= img.shape[0]:
            crd[3] = img.shape[0]-1

    return img_sheared, crd