# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:29:19 2018
@author: A_lifePC

Hamada, Ito の血管剛性の高周波成分をLPFで取り除くプログラム(試作用)
各被験者データのディレクトリに変更後， data.xlsx　内のbetaにフィルタ処理
"""

# 【import】 ####################################################################
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import math

# 以下関数 ##################################################################### 

# fft
def fft(f,s,r):#f:データ、s:サンプリング周期、r:時間データ
    N_data = len(f)
    #fs=1000/s
    fs=1/s   
    N = 2**11    # FFTのサンプル数
    # 最低、N点ないとFFTできないので0.0を追加
    for i in range(N-N_data): f.append(0.0)
    #print('2のべき乗の長さに変更後',len(f))    
    Window1 = np.hamming(N)
#    Window2 = np.hanning(N)   
    f_w = Window1*f
        # 高速フーリエ変換
    F = np.fft.fft(f_w)/(N/2)
        # 直流成分の振幅を揃える
    #F[0] = F[0]/2
        # 振幅スペクトル
    amp = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in F]  
        # 位相スペクトル
    phase = [np.arctan2(int(c.imag), int(c.real)) for c in F]    
        # 周波数軸の値を計算 
    freq = np.fft.fftfreq(N,d=1.0/fs)
        # ナイキスト周波数の範囲内のデータのみ取り出し
    freq = freq[1:int(N/2)]
    amp = amp[1:int(N/2)]
    phase = phase[1:int(N/2)]
    
    return freq,amp


def createLPF(fc):
    """IIR版ローパスフィルタ、fc:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2
    b[0] = (4 * np.pi**2 * fc**2) / denom
    b[1] = (8 * np.pi**2 * fc**2) / denom
    b[2] = (4 * np.pi**2 * fc**2) / denom
    a[0] = 1.0 #コードでは使わないけど，1.0を代入
    a[1] = (8 * np.pi**2 * fc**2 - 2) / denom
    a[2] = (1 - (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2) / denom
    return a, b

def iir(x, a, b):
    """IIRフィルタをかける、x:入力信号、a, b:フィルタ係数"""
    y = [0.0] * len(x)  # フィルタの出力信号
    Q = len(a) - 1
    P = len(b) - 1
    for n in range(len(x)):
        for i in range(0, P + 1):
            if n - i >= 0:
                y[n] += b[i] * x[n - i]
        for j in range(1, Q + 1):
            if n - j >= 0:
                y[n] -= a[j] * y[n - j]
    return y


# 3次スプライン補完でリサンプリング
def resmp(time,data):
#    time = df_out['newTime']
#    data = df_out['β']
    
    f_CS = interp1d(time, data, kind='cubic')  # 3次スプライン補完
    timerange = range(math.ceil(time[0]),math.floor(max(time)))    #　時間軸を整数化
    # math.ceilは切り上げ、math.floorは切り捨て    

    time_resmp = pd.Series([i for i in timerange])
    data_resmp = pd.DataFrame(([f_CS(i) for i in timerange]))   # 整数時間に沿ってリサンプリング    
    
    return time_resmp, data_resmp

def LPF(fs, fc, data):    
    fc_analog = np.tan(fc * np.pi / fs) / (2 * np.pi) 
    a, b = createLPF(fc_analog)
    # 順方向フィルタ
    data_f           = iir(data, a, b)          
    data_f_rev       = list(reversed(data_f))   # 反転
    # 逆方向フィルタ
    data_f_rev_f     = iir(data_f_rev, a, b)    
    data_f_rev_f_rev = list(reversed(data_f_rev_f))
    #　フィルタ後、反転させて元の順に戻す。これがゼロ位相フィルタ後。
    
    data_LPF = data_f_rev_f_rev
    
    return data_LPF

# 標準化
def std(x):
    xmean = x.mean()
    xstd  = np.std(x)
    zscore = (x-xmean)/xstd    

    return zscore

# 【前処理】 ####################################################################

# データがあるパスに移動
Datapath = 'C:\\Users\\A_lifePC\\Documents\\Matsushita Koyata\\1 .研究用\\2 .実験結果\\20180899_不快臭被験者増加\\20181002_Ito\\β解析テンプレ'
# os.getcwd()

#ディレクトリ内のデータファイルのリスト
Subnames = os.listdir(Datapath)

# 出力フォルダ
Datapath2 = 'C:\\Users\\A_lifePC\\Documents\\Matsushita Koyata\\1 .研究用\\2 .実験結果\\20180899_不快臭被験者増加\\20181002_Ito\\血管剛性LPF用'
if not os.path.isdir(Datapath2+'\\'+'result_LPbeta'):
    os.mkdir(Datapath2 + '\\' + 'result_LPbeta')
Datapath_out = Datapath2 + '\\' + 'result_LPbeta'

# 読み込むファイル名
input_file_name = 'data.xlsx'

# 【メイン処理】 ##################################################################

#ディレクトリの変更
os.chdir(Datapath)

#　DataFrameとしてベータの解析結果をよみとる(ファイル名は data.xlsx)
#databook = pd.ExcelFile('data.xlsx')

# βは第19列
df_data = pd.read_excel('data.xlsx', usecols=[19])

 # フィルタ処理
fs = 1
fc = 0.15
df_out = LPF(fs, fc, df_data)