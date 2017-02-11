#!/usr/bin/env python
# -*- coding: utf-8 -*-



# Jidan Zhong 
# 2017- Jan-20

import cv2
import numpy as np

from os import listdir
from os.path import isfile, join
myimgpath = '/media/truecrypt1/Research/TN_proj/dl_proj/image'
mysegpath = '/media/truecrypt1/Research/TN_proj/dl_proj/label'
myimgpathnew = '/media/truecrypt1/Research/TN_proj/dl_proj/imageds'
mysegpathnew = '/media/truecrypt1/Research/TN_proj/dl_proj/labelds'

imgfiles = [f for f in listdir(myimgpath) if isfile(join(myimgpath, f))]
segfiles=[f[:-13]+'_seg'+f[-13:] for f in imgfiles ]
fctr=0.5
for i in range(1,4032): #4032
    img = cv2.imread(join(myimgpath, imgfiles[i]))
    res = cv2.resize(img,None,fx=fctr, fy=fctr, interpolation = cv2.INTER_LINEAR)
    crop_img = res[17:112, 17:112]
    cv2.imwrite(join(myimgpathnew, imgfiles[i]), crop_img);

    img1 = cv2.imread(join(mysegpath, segfiles[i]))
    res1 = cv2.resize(img1,None,fx=fctr, fy=fctr, interpolation = cv2.INTER_LINEAR)
    ret1,th1 = cv2.threshold(res1,127,255,cv2.THRESH_BINARY)
    crop_img1 = th1[17:112, 17:112] # remove 16 slices from each of the four sides 
    cv2.imwrite(join(mysegpathnew, segfiles[i]), crop_img1);



# #########################################################
# #########################################################
# # check images

# import matplotlib.pyplot as plt

# plt.subplot(121),plt.imshow(crop_img),plt.title('Input')
# plt.subplot(122),plt.imshow(crop_img1),plt.title('Output')
# plt.show()
