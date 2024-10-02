import cv2

import cv2
import numpy as np
import pylab as pl
from PIL import Image
from PIL import ImageFilter
import os
import glob
#import features

# ROIpath = 'C:/Users/Architect Iqra Ayoub/PycharmProjects/MyfirstPro/Project/roi/'
# GaborPath = r'C:/Users/Architect Iqra Ayoub/PycharmProjects/MyfirstPro/Project/gabor/'
#

def Gabor_h(i,ROIpath,R_EyeBrowPath,shotname,GaborPath,SheetPath,FeaturesPath):
    #def Gabor_h_re(i, Right_Eyepath, shotname, GaborPath, SheetPath, FeaturesPath):

    cur_dir2 = 'C:/Users/Architect Iqra Ayoub/PycharmProjects/MyfirstPro/Project/gabor/'#path to store Gabor features
    if not os.path.exists(GaborPath):
        os.mkdir(os.path.join(GaborPath))
    Gaborpath = os.path.join(GaborPath, shotname)
    if not os.path.exists(Gaborpath):
        os.mkdir(Gaborpath)

    img=cv2.imread(ROIpath)                          # Loading color picture
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Change color picture into gray picture
    imgGray_f = np.array(imgGray,dtype=np.float64)   # Change data type of picture
    imgGray_f /=255.

    wavelenth = 15
    orentation = 90
    kernel_size = 12    #12
    sig =5                           #bandwidth
    gm = 0.5
    ps = 0.0
    th = orentation*np.pi/180
        #th=0.14
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sig, th,wavelenth,gm,ps)
    kernelimg=kernel /2 + 0.5
    dest = cv2.filter2D(imgGray_f, cv2.CV_32F, kernel)#CV_32F
    Gabor_Path = Gaborpath + '/'+str('%02d' % i) + '.jpg'
    cv2.imwrite(Gabor_Path, np.power(dest, 2))

    # used from yan's code
    #features.Features(i, shotname, Gabor_Path, SheetPath, FeaturesPath)
    # return i,shotname,Gabor_Path