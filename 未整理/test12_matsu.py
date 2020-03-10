# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

やること
フィルタかける
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
from scipy import fftpack
import matplotlib.pyplot as plt

# 【前処理】 ####################################################################
# 快臭データ
DATAPATH_P = "C:/Users/A_lifePC/Desktop/ファイル読み込み用/2018_12/快臭"
FILENAME_LIST_P_BIO = ["subA_ple.csv",
                       "subB_ple.csv",
                       "subC_ple.csv",
                       "subD_ple.csv",
                       "subE_ple.csv",
                       "subF_ple.csv"]


# 【関数定義】 ##################################################################

# 【メイン処理】 ##################################################################
# data読み取り
os.chdir(DATAPATH_P)
# encoding="cp932"を付けないと文字コードエラー
df_subA_bio = pd.read_csv(FILENAME_LIST_P_BIO[0], encoding="cp932")

# graph check
#plt.figure(1)
#plt.plot(df_subA_bio["Time"], df_subA_bio["β"])
#plt.show()


beta = df_subA_bio["β"].values
#beta = sp.stats.zscore(beta) # 標準化(scipyではnan入りを標準化できない)
beta = (beta - np.nanmean(beta))/np.nanstd(beta, ddof=1) # 標準化
n = len(beta)
# FFT処理と周波数スケールの作成
betaf = fftpack.fft(beta)/(n/2)
freq = fftpack.fftfreq(n)

# フィルタ処理
# カットオフ周波数以上に対応するデータを0とする
fs = 0.05 # カットオフ周波数
betaf2 = np.copy(betaf)
betaf2[(freq > fs)] = 0
betaf2[(freq < 0)] = 0

# 逆FFT処理
beta2 = np.real(fftpack.ifft(betaf2)*n)

# プロット
plt.figure(1)
plt.subplot(211)
plt.plot(freq[1:n//2], np.abs(betaf[1:n//2]))
plt.ylabel("Amplitude")
plt.axis("tight")
plt.subplot(212)
plt.plot(freq[1:n//2], np.abs(betaf2[1:n//2]))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.axis("tight")

plt.figure(2)
plt.plot(beta, "b", label="original")
plt.plot(beta2, "r", linewidth=2, label="filtered")
plt.axis("tight")
plt.legend(loc="upper right")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()