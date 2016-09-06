# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from alignment import findTransform
import shadow
import numpy as np
import cv2
    
def prepareImg(pathToImages,img_name):
    img_path = pathToImages + img_name
    img = cv2.imread(img_path)
    s = img.shape
    s_small = (int(s[1]/4),int(s[0]/4))
    s_small = (774,582)
    img_small = cv2.resize(img,s_small)
    img_grsmall = cv2.cvtColor(img_small,cv2.COLOR_RGB2GRAY)
    return img_small, img_grsmall
    
def warpImg(img_small, img_grsmall, base_img_grsmall):
    s = img_grsmall.shape
    s_small = (s[1],s[0])
    img_rsmall = img_small[:,:,0]
    img_gsmall = img_small[:,:,1]
    img_bsmall = img_small[:,:,2]
    M = findTransform(img_grsmall,base_img_grsmall,drawFlag=1)
    img_rwarp_small = cv2.warpPerspective(img_rsmall,M,s_small)
    img_gwarp_small = cv2.warpPerspective(img_gsmall,M,s_small)
    img_bwarp_small = cv2.warpPerspective(img_bsmall,M,s_small)
    img_cwarp_small = np.stack((img_rwarp_small,img_gwarp_small,img_bwarp_small),2)
    return img_cwarp_small

pathToImages = "..//train_sm//"

img1_name = "set5_1.jpeg"
img2_name = "set5_2.jpeg"
img3_name = "set5_3.jpeg"
img4_name = "set5_4.jpeg"
img5_name = "set5_5.jpeg"

img1_small, img1_grsmall = prepareImg(pathToImages,img1_name)

img1_c3_b, img1_S, img1_V, img1_V_edge = shadow.preprocess(img1_small)
img1_shadow_inds = shadow.seedDetect(img1_c3_b,img1_S,img1_V,5,0.35,0.02)
