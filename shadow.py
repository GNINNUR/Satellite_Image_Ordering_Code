# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:58:47 2016

@author: Sean
"""

import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.filters import laplace
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
    V_edge = laplace(V)
    return c3_b, S, V, V_edge
    
def window(i,j,n):
    w_inds = np.empty((0,2))
    for ii in np.arange(-n,n):
        for jj in np.arange(-n,n):
            w_inds = np.append(w_inds, [[i+ii,j+jj]],0)
    return w_inds
    
def seedDetect(c3_b,S,V,k,Tv,Ts):
#    k = 5 # must be odd
#    Tv = 0.35
#    Ts = 0.02
    n = (k-1)/2
    c3_mean = np.mean(c3_b)
    kernel = np.ones((k,k))
    c3_box = ndi.generic_filter(c3_b,f,footprint=kernel,mode='constant',cval=0)
    V_mean = ndi.generic_filter(V,f,footprint=kernel,mode='constant',cval=0)
    S_mean = ndi.generic_filter(S,f,footprint=kernel,mode='constant',cval=0)
    c3_gt_mean = c3_box > c3_mean
    V_lt_thresh = V_mean < Tv
    S_gt_thresh = S_mean > Ts
    coordinates = peak_local_max(c3_b, min_distance=k)
    seeds = []
    for c in coordinates:
        i = c[0]
        j = c[1]
        if c3_gt_mean[i,j]:
            if V_lt_thresh[i,j]:
                if S_gt_thresh[i,j]:
                    seed_inds = window(i,j,n)
                    seeds.append(seed_inds)
                    
    return seeds
    
def growRegion(c3,V,S,V_edge,shadow_inds,seeds,d0,k,Te,Tv,Ts):
    n = (k-1)/2
    c3_vals = np.array([c3[c[0],c[1]] for c in shadow_inds])
    c3_mean = c3_vals.mean()
    c3_std = c3_vals.std()
    border_shadow_inds = shadow_inds
    change_flag = 0
    for c in shadow_inds:
        i = c[0]
        j = c[1]
        border_pxls = window(i,j,n)
        for p in border_pxls:
            if p in shadow_inds:
                continue
            for s in seeds:
                if p in s:
                    continue
            if (np.abs(c3[i,j]-c3_mean)/c3_std) < d0:
                if V_edge[i,j] < Te:
                    if V[i,j] < Tv:
                        if S[i,j] > Ts:
                            change_flag = 1
                            border_shadow_inds = np.append(border_shadow_inds,[p],0) 
    new_shadow_inds = border_shadow_inds
    if change_flag:
        new_shadow_inds = growRegion(c3,V,S,V_edge,border_shadow_inds,seeds,d0,k,Te,Tv,Ts)
    return new_shadow_inds
                            
    