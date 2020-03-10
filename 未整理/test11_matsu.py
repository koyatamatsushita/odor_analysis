# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

やること
周波数解析
サンプリング周波数を1Hzとしている
hamming窓を用いて前処理しようかな

めも：セッション毎に分けて解析したほうがいいかも
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
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
beta = sp.stats.zscore(beta) # 標準化
window = np.hamming(len(beta)) # ハミング窓関数

windowedBeta = beta * window

plt.figure()
plt.plot(beta)
plt.plot(windowedBeta)
plt.plot(window)
plt.show()

F = np.fft.fft(windowedBeta) # 変換結果
freq = np.fft.fftfreq(len(windowedBeta), d=1) # 周波数
"""
fig, ax = plt.subplots(nrows=3, sharex=True)
ax[0].plot(F.real, label="Real part")
ax[0].legend()
ax[1].plot(F.imag, label="Imaginary part")
ax[1].legend()
ax[2].plot(freq, label="Frequency")
ax[2].legend()
ax[2].set_xlabel("Number of data")
plt.show()
"""
Amp = np.abs(F/(len(windowedBeta)/2)) # 振幅

fig, ax = plt.subplots()
ax.plot(freq[1:int(len(windowedBeta)/2)], Amp[1:int(len(windowedBeta)/2)])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()
