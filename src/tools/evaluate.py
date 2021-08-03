"""
評価指標を算出する
"""
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal,interpolate
from . import visualize
from ..calc_hrv import preprocessing
from scipy.sparse import spdiags

def CalcFreqTimeHRV(rpeaks,rri=None, tw=120,overlap=30,fps=30):
    """Plot spectrogram

    Parameters
    ----------
    rpeaks [ms]
    rri [ms]
    """
    rpeaks = rpeaks*0.001
    # 信号を補間する
    sample_rate = 4 # 補間するサンプリングレート
    # 3次のスプライン補間
    rri_spline = interpolate.interp1d(rpeaks, rri, 'cubic')
    t_interpol = np.arange(rpeaks[0], rpeaks[-1], 1./sample_rate)
    rri_interpol = rri_spline(t_interpol)

    # トレンドを除去
    filtered_rri_interpol = detrend(rri_interpol,Lambda=500)
    
    # 短時間フーリエ解析
    nfft= int(tw*sample_rate)
    f, t, Sxx = signal.spectrogram(filtered_rri_interpol, fs=sample_rate, nperseg=nfft,window='hamming',
                                   noverlap=int(nfft-overlap*sample_rate), scaling="density")
    df = None
    for i,t in enumerate(t):
        segment_bio_report = {}
        # 生体指標を算出
        parameter =  CalcParameter(f, Sxx[:,i])
        segment_bio_report.update({'Time':t})
        segment_bio_report.update(parameter)

        if df is None:
            df = pd.DataFrame([], columns=segment_bio_report.keys())
        df =  pd.concat([df, pd.DataFrame(segment_bio_report , index=[t])])

    return df

def CalcParameter(f, power):
    """
    HRVの生体指標を算出

    Keys:'vlf'	Very low frequency		(default: (0.003Hz, 0.04Hz))
         'lf'	Low frequency			(default: (0.04Hz - 0.15Hz))
         'hf'	High frequency			(default: (0.15Hz - 0.4Hz))
    """
    # 周波数帯を設定
    vlf_f = [0.,0.04]
    lf_f = [0.04,0.15]
    hf_f = [0.15,0.4]

    # インデックスを取得
    vlf_i = np.where((f > vlf_f[0]) & (f <= vlf_f[1]),True,False)
    lf_i = np.where((f > lf_f[0]) & (f <= lf_f[1]),True,False)
    hf_i = np.where((f > hf_f[0]) & (f <= hf_f[1]),True,False)
    total_i = np.where(f <= hf_f[1],True,False)
    
    # パワースペクトルを算出
    df = (f[1] - f[0])
    vlf_power = np.sum(power[vlf_i]) * df
    lf_power = np.sum(power[lf_i]) * df
    hf_power = np.sum(power[hf_i]) * df
    abs_powers = (vlf_power, lf_power, hf_power) 
    total_power = np.sum(abs_powers)

    # Normalized powers
    norms = tuple([100 * x / (lf_power + hf_power) for x in [lf_power, hf_power]])
    
    # LF/HF Ratio
    lfhf_ratio = lf_power/hf_power
	# 返り値の作成
    bio_parameters = {"LF_Norm":norms[0],
                      "HF_Norm":norms[1],
                      "LF_ABS":abs_powers[1],
                      "HF_ABS":abs_powers[2],
                      "LFHFratio": lfhf_ratio}
    return bio_parameters

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

def CalcSNR(ppg, HR_F=None, fs=30, nfft=1024):
    """
    CHROM参照
    SNRを算出する
    SNR が大きいと脈拍の影響が大きく，
    SNRが小さいとノイズの影響が大きい
    """
    freq, power = signal.welch(ppg, fs, nfft=nfft, detrend="constant",
                                     scaling="spectrum", window="hamming")

    FMask = (freq >= 0.5)&(freq <= 4) # SNR算出範囲を指定
    FMask2 = (freq >= 0.5)&(freq <= 1.5) # 心拍数特定範囲を指定
    
    if HR_F is None:
        power_sub = power[FMask2]
        HR_F = freq[FMask2][np.argmax(power_sub)]
    # 0.2Hz帯
    GTMask1 = (freq >= HR_F-0.1) & (freq <= HR_F+0.1)
    GTMask2 = (freq >= HR_F*2-0.2) & (freq <= HR_F*2+0.2)
    SPower = np.sum(power[GTMask1 | GTMask2])
    
    AllPower = np.sum(power[FMask])
    SNR = 10*np.log10((SPower)**2/(AllPower-SPower)**2)
    return {"HR":HR_F,"SNR":SNR}

def CalcTimeSNR(ppg,fs,tw=10):
    """
    Calculate Frequency domain heart rate
    using DFT,
    segment = 10s
    return HR[bpm]
    """

    # デフォルトは50%オーバーラップ
    nfft= int(fs*tw)
    freq, t, Sxx = signal.spectrogram(ppg, fs=fs, scaling="spectrum",nperseg=nfft,noverlap=nfft//2)
    FMask = (freq >= 0.5)&(freq <= 4) # SNR算出範囲を指定
    FMask2 = (freq >= 0.5)&(freq <= 2) # 心拍数特定範囲を指定

    plt.pcolormesh(t, freq*60, Sxx,cmap="jet",shading='gouraud')
    plt.ylabel('Frequency [BPM]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 250)
    plt.show()

    result = np.array([[],[]])
    for Sxx_i in Sxx.T:
        # Calculate HR
        Sxx_sub = Sxx_i[FMask2]
        HR_F = freq[FMask2][np.argmax(Sxx_sub)] # 心拍数を決定
        # Calculate SN-Ratio
        # 0.2Hz帯
        GTMask1 = (freq >= HR_F-0.1) & (freq <= HR_F+0.1)
        GTMask2 = (freq >= HR_F*2-0.2) & (freq <= HR_F*2+0.2)
        SPower = np.sum(Sxx_i[GTMask1 | GTMask2])
        AllPower = np.sum(Sxx_i[FMask])
        SNR = 10*np.log10((SPower)**2/(AllPower-SPower)**2)
        result = np.concatenate((result,np.array([[HR_F],[SNR]])),axis=1)
        plt.plot(freq*60,Sxx_i)
        plt.xlim(0, 250)
        plt.axvspan(60*(HR_F-0.1), 60*(HR_F+0.1), color = "coral", alpha=0.2)
        plt.axvspan(60*(2*HR_F-0.2), 60*(2*HR_F+0.2), color = "coral", alpha=0.2)
        plt.title(SNR)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    result = np.concatenate((t.reshape(-1,1),result.T),axis=1)

    return result

def CalcEvalRRI(ref_rri, est_rri):
    corr = np.corrcoef(est_rri, ref_rri)[0, 1]
    x = 0.5*(est_rri + ref_rri)
    y = (est_rri - ref_rri)
    mae = np.mean(abs(y))
    rmse = np.sqrt(np.mean(y**2))
    result = {"corr":corr,"mae":mae,"rmse":rmse}
    return result

def CalcPSD(rri_peaks=None, rri=None, nfft=2**8,keyword=""):
    """
    PSDを出力
    rri_peaks [ms]
    """
    rri_peaks *= 0.001
    rri *= 0.001
    sample_rate = 4
    
    # 3次のスプライン補間
    rri_spline = interpolate.interp1d(rri_peaks, rri, 'cubic')
    t_interpol = np.arange(rri_peaks[0], rri_peaks[-1], 1./sample_rate)
    rri_interpol = rri_spline(t_interpol)
    frequencies, powers  = signal.welch(x=rri_interpol, fs=sample_rate, window='hamming',
                                        detrend="constant",	nperseg=len(rri_interpol),
                                        nfft=len(rri_interpol), scaling='density')
    freqdf = (frequencies[1] - frequencies[0])# Compute frequency resolution
    LF = np.sum(powers[(frequencies>=0.05) & (frequencies<0.15)]) * freqdf
    HF = np.sum(powers[(frequencies>0.15) & (frequencies<=0.40)]) * freqdf
    VLF = np.sum(powers[(frequencies>0.0033) & (frequencies<=0.05)]) * freqdf
    print("Result :LF={:2f}, HF={:2f}, LF/HF={:2f}".format(LF, HF, LF/HF))
    return {f'{keyword}VLF_abs':VLF,
            f'{keyword}LF_abs':LF,
            f'{keyword}HF_abs':HF,
            f'{keyword}LFHFratio':LF/HF}

def CalcMissPeaks(est_rpeaks=None, 
                   ref_rpeaks=None,
                   threshold=0.25):
    """
    リファレンスのピークと，推定したピーク値を比較
    ピーク検出に失敗する割合を算出する
    ----------
	ref_rpeaks, est_rpeaks : array
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
    error_flag=0
    # ピーク時間のずれを補正
    # 最初のピーク位置でECGとPPGの位相遅れに対処する
    t_first = np.maximum(est_rpeaks[0],ref_rpeaks[0])
    est_rpeaks = est_rpeaks[(t_first<=est_rpeaks)]
    ref_rpeaks = ref_rpeaks[(t_first<=ref_rpeaks)]
    est_rpeaks = est_rpeaks - est_rpeaks[0]
    ref_rpeaks = ref_rpeaks - ref_rpeaks[0]


    # RRIを取得
    est_rri = est_rpeaks[1:]-est_rpeaks[:-1]
    est_rpeaks = est_rpeaks[1:]
    ref_rri = ref_rpeaks[1:]-ref_rpeaks[:-1]
    ref_rpeaks = ref_rpeaks[1:]
    input_peaknum = ref_rri.size
    # print("Input  REF RRI:{}, EST RRI:{}".format(ref_rri.size, est_rri.size))

    if est_rpeaks.size != ref_rpeaks.size:
        # Estimate peaks内で閾値より大きく外れたデータを削除
        median_rri = signal.medfilt(est_rri, 5)# median filter
        detrend_est_rri = est_rri - median_rri
        index_outlier = np.where(np.abs(detrend_est_rri) > (threshold*1000))[0]
        # print("{} point detected".format(index_outlier.size))
        if index_outlier.size > 0:
            flag = np.ones(len(est_rri), dtype=bool)
            flag[index_outlier.tolist()] = False
            est_rpeaks = est_rpeaks[flag]
            est_rri = est_rri[flag]
            
            # リファレンスと比較して，大きく外れたデータを検出
            ref_index = []
            for i,i_rpeak in enumerate(ref_rpeaks):
                # リスト要素と対象値の差分を計算し最小値のインデックスを取得
                idx = np.abs(est_rpeaks-i_rpeak).argmin()
                if np.abs(est_rpeaks[idx]-i_rpeak) <= (0.50*1000):
                    ref_index.append(i)
            ref_rpeaks = ref_rpeaks[ref_index]
            ref_rri = ref_rri[ref_index]
        
        # さらにrpeaksの数が合わない場合
        if ref_rri.size != est_rri.size:
            # 最後の配列を削除する
            # なぜ合わないかは不明
            length = min(ref_rri.size,est_rri.size)
            ref_rri = ref_rri[:int(length)]
            est_rri = est_rri[:int(length)]
            error_flag = True

            
    # print("Output REF RRI:{}, EST RRI:{}".format(ref_rri.size, est_rri.size))
    
    error_rate = (input_peaknum-ref_rri.size)/input_peaknum
    # print("Error Rate: {}%".format(100*error_rate))
    return ref_rri,est_rri,error_rate,error_flag

if __name__ == "__main__":
    path = r"D:\rPPGDataset\Analysis\luminance\shizuya\2021-01-05 18-45-41.248194 Front And Celling 700lux rPPG Signals.csv"
    rppg_signal = np.loadtxt(path,delimiter=",")[:,0]
    fs = 30
    
