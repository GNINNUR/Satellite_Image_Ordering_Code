# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:58:47 2016

@author: Sean
"""

import numpy as np
import cv2
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def f(values):
    return values.mean()

def RGB2C3(img):
    s = img.shape
    c3 = np.zeros((s[0],s[1]))
    for i in range(s[0]):
        for j in range(s[1]):
            c3[i,j] = np.arctan(img[i,j,0]/np.max(img[i,j,1:]))
            if c3[i,j] == np.nan:
                c3[i,j] = 0
    return c3
    
def preprocess(img_RGB):
    c3 = RGB2C3(img_RGB)
    hsv = cv2.cvtColor(img_RGB,cv2.COLOR_RGB2HSV)
    S = hsv[:,:,1]
    V = hsv[:,:,2]
    c3_b = cv2.blur(c3,(3,3))
    V_edge = cv2.Laplacian(V)
    return c3_b, S, V, V_edge
    
def seedDetect(c3_b,S,V,k,Tv,Ts):
#    k = 5
#    Tv = 0.35
#    Ts = 0.02
    c3_mean = np.mean(c3_b)
    kernel = np.ones((k,k))
    c3_mean = ndi.generic_filter(c3_b,f,kernel,mode='constant',cval=0)
    V_mean = ndi.generic_filter(V,f,kernel,mode='constant',cval=0)
    S_mean = ndi.generic_filter(S,f,kernel,mode='constant',cval=0)
    c3_gt_mean = c3_mean > c3_mean
    V_gt_mean = V_mean > Tv
    S_gt_mean = S_mean > Ts
    coordinates = peak_local_max(c3_b, min_distance=k)
    shadow_inds = np.empty((0,2))
    for (i,j) in coordinates:
        if c3_gt_mean[i,j]:
            if V_gt_mean[i,j]:
                if S_gt_mean[i,j]:
                    shadow_inds = np.append(shadow_inds, [[i,j]])
                    shadow_inds = np.append(shadow_inds, [[i+1,j]])
                    shadow_inds = np.append(shadow_inds, [[i-1,j]])
                    shadow_inds = np.append(shadow_inds, [[i,j+1]])
                    shadow_inds = np.append(shadow_inds, [[i,j-1]])
                    shadow_inds = np.append(shadow_inds, [[i+1,j+1]])
                    shadow_inds = np.append(shadow_inds, [[i-1,j-1]])
                    shadow_inds = np.append(shadow_inds, [[i+1,j-1]])
                    shadow_inds = np.append(shadow_inds, [[i-1,j+1]])
                    
    return shadow_inds
            
    