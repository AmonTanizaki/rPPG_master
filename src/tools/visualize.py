"""
可視化用
"""
# coding: utf-8
import numpy as np
from scipy import signal,interpolate
from scipy.sparse import spdiags
from ..calc_hrv import preprocessing
from ..calc_hrv import peak_detector
import matplotlib.pyplot as plt

import matplotlib as mpl
import seaborn as sns
sns.set(font_scale=8/6)

def plot_SNR(ppg, hr=None, fs=30,text=None):
    """
    CHROM参照
    SNRを算出する
    SNR が大きいと脈拍の影響が大きく，
    SNRが小さいとノイズの影響が大きい
    """
    NyquistF = fs/2;
    FResBPM = 0.5 # パワースペクトルの解像度（bpm）
    N = (60*2*NyquistF)/FResBPM

    freq, power = signal.welch(ppg, fs, nfft=5096*8, detrend="constant",
                                     scaling="spectrum", window="hamming")
    # peak hr
    if hr is not None:
        HR_F = hr/60
    else:
        HR_F = freq[np.argmax(power)]
        print(HR_F)
    # 0.2Hz帯
    GTMask1 = (freq >= HR_F-0.1) & (freq <= HR_F+0.1)
    GTMask2 = (freq >= HR_F*2-0.2) & (freq <= HR_F*2+0.2)
    SPower = np.sum(power[GTMask1 | GTMask2])
    FMask2 = (freq >= 0.5)&(freq <= 4)
    AllPower = np.sum(power[FMask2])
    SNR = 10*np.log10((SPower)**2/(AllPower-SPower)**2)
    # normalize
    power_norm = (power-np.min(power))/(np.max(power)-np.min(power))
    plt.figure()
    plt.plot(freq*60, power_norm,label=text)
    # plt.axvspan(60*(HR_F-0.1), 60*(HR_F+0.1), color = "coral", alpha=0.2)
    # plt.axvspan(60*(2*HR_F-0.2), 60*(2*HR_F+0.2), color = "coral", alpha=0.2)
    plt.xlabel("Frequency [bpm]")
    plt.ylabel("Normalized Amplitude [-]")
    plt.xlim(0, 250)
    print(HR_F, SNR)
    plt.title("freq HR: {:.2f}  SNR: {:.2f}".format(HR_F, SNR))

def plot_PPGspec(ppg, fs=None, tw=10):
    """
    Plot spectrogram
    """
    nfft= int(fs*tw)
    f, t, Sxx = signal.spectrogram(ppg, fs=fs, scaling="spectrum",nperseg=nfft,noverlap=nfft//2)

    # FMask2 = (freq >= 0.5)&(freq <= 4)
    # if HR_F is None:
    #     power_sub = power[FMask2]
    #     HR_F = freq[FMask2][np.argmax(power_sub)]
    # # 0.2Hz帯
    # GTMask1 = (freq >= HR_F-0.1) & (freq <= HR_F+0.1)
    # GTMask2 = (freq >= HR_F*2-0.2) & (freq <= HR_F*2+0.2)
    # SPower = np.sum(power[GTMask1 | GTMask2])
    
    # AllPower = np.sum(power[FMask2])
    # SNR = 10*np.log10((SPower)**2/(AllPower-SPower)**2)
    plt.pcolormesh(t, f*60, Sxx,cmap="jet",shading='gouraud')
    plt.ylabel('Frequency [BPM]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 250)
    plt.show()

# トレンド除去
def detrend(rri, Lambda):
    """applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    
    Parameters
    ----------
    rri: numpy.ndarray
        The rri where you want to remove the trend. 
        ***  This rri needs resampling  ***
    Lambda: int
        The smoothing parameter.

    Returns
    ------- 
    filtered_rri: numpy.ndarray
        The detrended rri.
    
    """
    rri_length = rri.shape[0]

    # observation matrix
    H = np.identity(rri_length) 

    # second-order difference matrix
    ones = np.ones(rri_length)
    minus_twos = -2*np.ones(rri_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (rri_length-2), rri_length).toarray()
    filtered_rri = np.dot((H - np.linalg.inv(H + (Lambda**2) * np.dot(D.T, D))), rri)
    return filtered_rri

def plot_HRVspec(rpeaks,rri=None, tw=120,title=None):
    """
    rpeaks [ms]
    rri [ms]
    Plot spectrogram
    """
    rpeaks *= 0.001
    rri *= 0.001

    # 信号を補間する
    sample_rate = 4 # 補間するサンプリングレート
    # 3次のスプライン補間
    rri_spline = interpolate.interp1d(rpeaks, rri, 'cubic')
    t_interpol = np.arange(rpeaks[0], rpeaks[-1], 1./sample_rate)
    rri_interpol = rri_spline(t_interpol)

    filtered_rri_interpol = detrend(rri_interpol,Lambda=500)

    nfft= int(tw*sample_rate)
    f, t, Sxx = signal.spectrogram(filtered_rri_interpol, fs=sample_rate, nperseg=nfft,
                                   noverlap=nfft//2, scaling="spectrum")

    plt.pcolormesh(t, f, Sxx, shading='gouraud',cmap="jet")
    if title is not None:
        plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 0.40)
    plt.show()
    return f, t, Sxx

def plot_BlandAltman(ref_rri,est_rri):
    """
    Plot Bland Altman
    --------------------------
    ref_rri:refrence rri signals[ms]
    est_rri:estiomated rri signals[ms]
    """
    corr = np.corrcoef(est_rri, ref_rri)[0, 1]
    x = 0.5*(est_rri + ref_rri)
    y = (est_rri - ref_rri)
    mae = np.mean(abs(y))
    rmse = np.sqrt(np.mean(y**2))
    sygma = np.std(y)
    print("Result index: MAE={} RMSE={}".format(mae, rmse))
    plt.figure(figsize=(8,6))
    plt.scatter(x, y)
    plt.axhline(sygma*1.96,label="+1.96σ")
    plt.axhline(-sygma*1.96,label="-1.96σ")
    plt.axhline(np.mean(y),label="mean", color='black')
    plt.xlabel("(Estimate+Reference)/2 [ms]")
    plt.ylabel("Estimate-Reference [ms]")
    plt.title("Bland Altman Plot\nMAE={:.2f}, RMSE={:.2f}, CORR={:.2f}".format(mae, rmse, corr))
    plt.legend()
    plt.show()

def plot_HRVpsd(rri_peaks=None, rri=None, label=None,nfft=2**12):
    """
    PSDを出力
    rri_peaks [s]
    """
    #rri_peaks *= 0.001
    rri *= 0.001
    sample_rate = 4
    if rri is None:
        rri = np.diff(rri_peaks)
        rri_peaks = rri_peaks[1:] - rri_peaks[1]

    # 3次のスプライン補間
    rri_spline = interpolate.interp1d(rri_peaks, rri, 'cubic')
    t_interpol = np.arange(rri_peaks[0], rri_peaks[-1], 1./sample_rate)
    rri_interpol = rri_spline(t_interpol)
    frequencies, powers  = signal.welch(x=rri_interpol, fs=sample_rate, window='hamming',
                                        detrend="constant",	nperseg=nfft,
                                        nfft=nfft, scaling='density')
    LF = np.sum(powers[(frequencies>=0.05) & (frequencies<0.15)]) * 0.10
    HF = np.sum(powers[(frequencies>0.15) & (frequencies<=0.40)]) * 0.25
    print("Result :LF={:2f}, HF={:2f}, LF/HF={:2f}".format(LF, HF, LF/HF))
    plt.plot(frequencies, powers, label=label)
    plt.axvline(x=.05, color='r')
    plt.axvline(x=0.15, color='r')
    plt.axvline(x=0.40, color='r')
    plt.xlim(0,.5)
    plt.xlabel("frequency[Hz]")
    plt.ylabel("PSD[s^2/Hz]")