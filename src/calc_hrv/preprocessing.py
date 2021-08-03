"""
RGB値の前処理を担当する
前処理には，BPFが含まれる
"""
# coding: utf-8
import numpy as np
from scipy import signal
from scipy import interpolate

def ButterFilter(data, lowcut, highcut, fs, order=3):
    """
    Butter Band pass filter
    doc:
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
     https://www.it-swarm.dev/ja/python/scipysignalbutter%E3%81%A7%E3%83%90%E3%83%B3%E3%83%89%E3%83%91%E3%82%B9%E3%83%90%E3%82%BF%E3%83%BC%E3%83%AF%E3%83%BC%E3%82%B9%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF%E3%83%BC%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95/1067792786/
    scipy:0.4.1verと引数が変わっているようなので注意
    ・最小の方は少し振幅が小さくなる
    ・SOSの方が安定しているそう
    """
    detrend = data - np.mean(data)
    sos = ButterBandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfilt(sos, detrend)
    return -y

def ButterBandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def MovingAve(data, num=10,detrend=True):
    """
    移動平均
    """
    if detrend:
        data = data-np.mean(data)
    weight = np.ones(num)/num
    convolve_data = np.convolve(data, weight, mode='same')
    return data-convolve_data

def rgb_resample(rgb_signals, ts, fs=100):
    """
    一様にサンプリングされていない信号の目的のレートへのリサンプリング
    """                                
    rgb_signals_interpol = interpolate.interp1d(ts, rgb_signals, "linear",axis=0)
    t_interpol = np.arange(ts[0], ts[-1], 1./fs)
    rgb_signals_n = rgb_signals_interpol(t_interpol)
    return rgb_signals_n
    

if __name__ == "__main__":
    pass