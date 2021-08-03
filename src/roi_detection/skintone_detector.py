"""
肌色検出アルゴリズム
"""
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

def SkinDetect(img):
    # HSVの肌色検出
    mask_hsv = SkinDetectHSV(img)
    # YCbCrの肌色検出
    mask_ycbcr = SkinDetectYCbCr(img)
    mask_all = cv2.bitwise_and(mask_ycbcr, mask_ycbcr, mask_hsv)
    # skinmask = cv2.bitwise_and(img, img, mask=mask_all)
    mask_all = ReduceNoise(mask_all)
    return mask_all

def SkinDetectTrack(img,path):
    """
    手動で閾値設定
    """
    YCbCr_MIN,YCbCr_MAX = np.load(path)

    # convert BGR to yCbCr    
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #mask ycbcr region
    mask_ycbcr = cv2.inRange(img_ycbcr, YCbCr_MIN, YCbCr_MAX)
    return mask_ycbcr

def SkinDetectHSV(img,auto=True):
    """
    RGB to HSV
    """
    HSV_MIN = np.array([0, 40, 0])
    HSV_MAX = np.array([25, 255, 255])
    # convert BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if auto:
        #大津の二値化
        auto_threshold,_ = cv2.threshold(img_hsv[:,:,0],0,255,cv2.THRESH_OTSU)
        HSV_MAX[0] =  auto_threshold

    #mask hsv region
    mask_hsv = cv2.inRange(img_hsv, HSV_MIN, HSV_MAX)

    return mask_hsv

def SkinDetectYCbCr(img):
    """
    RGB to YCbCr
    """
    YCbCr_MIN = np.array([0, 138, 67])
    YCbCr_MAX = np.array([255, 173, 133])
    # convert BGR to yCbCr    
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #mask ycbcr region
    mask_ycbcr = cv2.inRange(img_ycbcr, YCbCr_MIN, YCbCr_MAX)

    return mask_ycbcr

def ReduceNoise(img):
    # ノイズ除去(膨張・収縮)
	# using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    skinMask = cv2.erode(img, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
	# ガウシアンフィルタにより，ノイズを抑える
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    return skinMask


if __name__ == "__main__":
    pass