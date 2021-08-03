"""
GREEN_VERKRUYSSE The Green-Channel Method from: 
Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445. DOI: 10.1364/OE.16.021434
"""
# coding: utf-8
import numpy as np
import pandas as pd
from ..calc_hrv import preprocessing

def GreenMethod(rgb_signals,fs,filter=True,LPF=0.7, HPF=2.5):
    """
    RPPG　Green法
    
    Parameters
    -------
    rgb_signals :array
       一連のRGB信号
    fs : int 
       フレームレート
    filter: bool,optional
        バンドパスフィルタによってノイズを除去するか選択
    LPF,HPF : float
        バンドパスの周波数帯を定義する

    Returns
    -------
    rppg: array
        RGB信号から抽出したG成分
    """
    green_sig = rgb_signals[:, 1]
    if filter:
        # Filter, Normalize
        rppg = preprocessing.ButterFilter(green_sig, LPF, HPF, fs)
    else:
        rppg = green_sig
    return rppg
