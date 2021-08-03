"""
実行ファイル
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Import local
from src.roi_detection.landmark_extractor import *
from src.pulse_extraction import *
from src.calc_hrv.peak_detector import *
from src.calc_hrv.preprocessing import *
from src.tools.openface import *
from src.tools import visualize
from src.tools.evaluate import *
from src.tools.skintrackbar import *
from src.tools.avitomp4 import *
from src.tools.hide_info import *

triming(r"D:\ICUDataset\1_Log\2021-07-12\out\00295.MTS",r"D:\ICUDataset\1_Log\2021-07-12\out\00295.mp4")

# #緒数を定義
# fps = 60 #フレームレート
# c_fps = 100 # 補間するフレームレート

# # 動画ファイルをmp4に変換
# # moviech(r"D:\ICUDataset\1_Log\2021-07-12\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 Cam.avi",
# #         r"D:\ICUDataset\1_Log\2021-07-12\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 Cam.mp4")

# # RGB信号を取り出す
# trackbar(r"D:\ICUDataset\1_Log\2021-07-12\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 Cam.avi",
#         r"D:\ICUDataset\1_Log\2021-07-12\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 SkinPath.npy")
# rgb_signals = MouseRoI(r"D:\ICUDataset\1_Log\2021-07-12\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 Cam.avi",
#         r"D:\ICUDataset\1_Log\2021-07-12\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 SkinPath.npy"
#         ,12,60)
# np.savetxt(r"D:\ICUDataset\1_Log\2021-07-12\out\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 RGB.csv",rgb_signals,delimiter=",")
# # rgb_signals = np.loadtxt(r"C:\Users\akito\Desktop\ICUDataset\1_Log\2021-07-12\out\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 RGB.csv",delimiter=",")

# # フレームレートのバラつきを補正
# timestamps = np.loadtxt(r"D:\ICUDataset\1_Log\2021-07-12\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 timestamp.csv",delimiter=",")
# rgb_signals = preprocessing.rgb_resample(rgb_signals,timestamps,fs=fps)

# # rPPG手法によって脈波信号を取り出す
# # rppg_sig = GreenMethod(rgb_signals,fps)
# # plt.plot(rppg_sig,label="g")
# rppg_sig = POSMethod(rgb_signals,fps)
# # plt.plot(rppg_sig,label="pos")
# # plt.legend()
# # plt.show()

# # ピーク検出
# rpeaks = RppgPeakDetection(rppg_sig, fps,fr=c_fps,show=True)
# # plt.plot(rpeaks[1:],rpeaks[1:]-rpeaks[:-1])

# # 外れ値の検出
# rpeaks, rri = OutlierDetect(rpeaks, threshold=0.25)
# HR = np.hstack((rpeaks.reshape(len(rpeaks),1),rri.reshape(len(rri),1)))

# # plt.plot(rpeaks, rri)
# # plt.show()

# # HRVのパラメータを算出
# # result = CalcPSD(rpeaks, rri, nfft=2**8,keyword="")
# # visualize.plot_HRVpsd(rpeaks, rri, nfft=2**8)
# # plt.show()

# np.savetxt(r"D:\ICUDataset\1_Log\2021-07-12\out\2021-07-12 16-08-57.143975 icu_91_satou_2021-07-12 HR.csv",HR,delimiter=",",header='rpeaks,rri')



