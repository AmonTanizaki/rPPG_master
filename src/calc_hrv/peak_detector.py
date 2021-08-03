"""
peak to peak detector
"""
# coding: utf-8
import numpy as np
from scipy import interpolate,signal
from . import preprocessing
from ..tools import evaluate
import matplotlib.pyplot as plt

def RppgPeakDetection(ppg,fs,fr=100, show=False, filter=False, range=.7):
    """
    rPPG peak検出

    Parameters
    -------
    ppg: array
        脈波信号
    fs: int 
        サンプリングレート
    fr: int 
        補間後のサンプリングレート
    show: bool,optional 
        ピーク検出の結果をグラフで表示
    filter: bool,optional
        ノイズ除去
    range: float,optional
        ピーク検出のパラメータ
    
    Returns
    -------
    rpeaks: array
        [ms] PPGのピーク時間
    """
    
    hr_f = evaluate.CalcSNR(ppg,fs=fs,nfft=1024)["HR"]

    if filter==True:
        # Moving Average
        ppg =  preprocessing.ButterFilter(ppg, 0.7, 2.5, fs)    
    
    # Resampling
    t_interpol, resamp_ppg = resampling(ppg, fs, fr)
    
    # Peak Detection 
    order=int((1/hr_f) * range * fr) # RRI[s] * range[%] * rate[hz] = サンプル数
    peak_indexes = signal.argrelmax(resamp_ppg,order=order)
    rpeaks = t_interpol[peak_indexes]

    if show:
        fig,axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(t_interpol, resamp_ppg)
        axes[0].set_title("Resample signal")
        for rpeak in rpeaks:
            axes[0].axvline(rpeak)
        axes[1].set_title("RRI signal")
        axes[1].plot(rpeaks[1:],rpeaks[1:]-rpeaks[:-1])
        plt.show()
    return rpeaks*1000 # [ms]

def OutlierDetect(rpeaks=None, threshold=0.25):
    """RRI時系列と平均値の差分を算出し，閾値を使って外れ値を取り除く
    Kubiosより参照
    !注意!補間するため，rpeaksからrriは算出できなくなる
    ----------
	rpeaks : array
		R-peak locations in [ms]
    threshold: float
        外れ値を検出するレベル．
        この値が高いほど外れ値と検出されるピークは多くなる
    ----------
    threshold level
    
    very low : 0.45sec
    low : 0.35sec
    medium : 0.25sec
    strong : 0.15sec
    very strong : .05sec
    """
    # RRIを取得
    rri = rpeaks[1:]-rpeaks[:-1]
    rpeaks = rpeaks[1:]-rpeaks[0]

    # median filter
    median_rri = signal.medfilt(rri, 5)
    detrend_rri = rri - median_rri

    # 閾値より大きく外れたデータを取得
    index_outlier = np.where(np.abs(detrend_rri) > (threshold*1000))[0]

    if index_outlier.size > 0:
        print("Outlier {} points detected".format(index_outlier.size))
        # 閾値を超えれば，スプライン関数で補間
        flag = np.ones(len(rri), dtype=bool)
        flag[index_outlier.tolist()] = False
        rri_spline = interpolate.interp1d(rpeaks[flag], rri[flag], 'cubic')
        try:
            rri_outlier = rri_spline(rpeaks[np.logical_not(flag)])
            rri[np.logical_not(flag)] = rri_outlier
        except:
            pass

    return rpeaks, rri

def resampling(rppg, fs, fr):
    """
    リサンプリング
    3次のスプライン補間
    """
    ts = np.arange(0, len(rppg)/fs, 1./fs)[:len(rppg)]
    rppg_interpol = interpolate.interp1d(ts, rppg, "cubic")
    t_interpol = np.arange(ts[0], ts[-1], 1./fr)
    resamp_rppg = rppg_interpol(t_interpol)
    return t_interpol, resamp_rppg
