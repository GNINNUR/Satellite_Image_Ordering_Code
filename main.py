# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from alignment import findTransform
import shadow
import numpy as np
import cv2
import os
    
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
img2_small, img2_grsmall = prepareImg(pathToImages,img2_name)
img3_small, img3_grsmall = prepareImg(pathToImages,img3_name)
img4_small, img4_grsmall = prepareImg(pathToImages,img4_name)
img5_small, img5_grsmall = prepareImg(pathToImages,img5_name)

img2_cwarp_small = warpImg(img2_small, img2_grsmall, img1_grsmall)
img3_cwarp_small = warpImg(img3_small, img3_grsmall, img1_grsmall)
img4_cwarp_small = warpImg(img4_small, img4_grsmall, img1_grsmall)
img5_cwarp_small = warpImg(img5_small, img5_grsmall, img1_grsmall)

img1_c3 = shadow.RGB2C3(img1_small)
img2_c3 = shadow.RGB2C3(img2_cwarp_small)
img3_c3 = shadow.RGB2C3(img3_cwarp_small)
img4_c3 = shadow.RGB2C3(img4_cwarp_small)
img5_c3 = shadow.RGB2C3(img5_cwarp_small)

img12_sub = img1_c3 - img2_c3
img13_sub = img1_c3 - img3_c3
img14_sub = img1_c3 - img4_c3
img15_sub = img1_c3 - img5_c3
img23_sub = img2_c3 - img3_c3
img24_sub = img2_c3 - img4_c3
img25_sub = img2_c3 - img5_c3
img34_sub = img3_c3 - img4_c3
img35_sub = img3_c3 - img5_c3
img45_sub = img4_c3 - img5_c3

img1_thresh = np.array(img1_c3 < 0.7,dtype=np.float)
img2_thresh = np.array(img2_c3 < 0.7,dtype=np.float)
img3_thresh = np.array(img3_c3 < 0.7,dtype=np.float)
img4_thresh = np.array(img4_c3 < 0.7,dtype=np.float)
img5_thresh = np.array(img5_c3 < 0.7,dtype=np.float)

#cv2.namedWindow("11")
#cv2.namedWindow("21")
#cv2.namedWindow("31")
#cv2.namedWindow("41")
#cv2.namedWindow("51")
#cv2.namedWindow("61")
#cv2.namedWindow("71")
#cv2.namedWindow("81")
#cv2.namedWindow("91")
#cv2.namedWindow("01")
cv2.namedWindow("1")
cv2.namedWindow("2")
cv2.namedWindow("3")
cv2.namedWindow("4")
cv2.namedWindow("5")
#cv2.imshow("11", img12_sub)
#cv2.imshow("21", img13_sub)
#cv2.imshow("31", img14_sub)
#cv2.imshow("41", img15_sub)
#cv2.imshow("51", img23_sub)
#cv2.imshow("61", img24_sub)
#cv2.imshow("71", img25_sub)
#cv2.imshow("81", img34_sub)
#cv2.imshow("91", img35_sub)
#cv2.imshow("01", img45_sub)
cv2.imshow("1", img1_c3)
cv2.imshow("2", img2_c3)
cv2.imshow("3", img3_c3)
cv2.imshow("4", img4_c3)
cv2.imshow("5", img5_c3)
#cv2.moveWindow("11", 10, 10) 
#cv2.moveWindow("21", 10, 10) 
#cv2.moveWindow("31", 10, 10) 
#cv2.moveWindow("41", 10, 10) 
#cv2.moveWindow("51", 10, 10) 
#cv2.moveWindow("61", 10, 10) 
#cv2.moveWindow("71", 10, 10) 
#cv2.moveWindow("81", 10, 10) 
#cv2.moveWindow("91", 10, 10) 
#cv2.moveWindow("01", 10, 10) 
cv2.moveWindow("1", 10, 10) 
cv2.moveWindow("2", 10, 10) 
cv2.moveWindow("3", 10, 10) 
cv2.moveWindow("4", 10, 10) 
cv2.moveWindow("5", 10, 10) 
cv2.waitKey(0)