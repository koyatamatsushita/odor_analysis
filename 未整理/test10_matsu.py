# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

やること
データ整理
フラグを管理して刺激時などを抽出する
微分する
セッション番号抽出
変動係数計算(変動係数は刺激中で計算することになるかな)
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib.pyplot as plt

# 【前処理】 ####################################################################
COLUMNS = ["Timing",
           "Step",
           "Time",
           "KanniKetteiKeisu",
           "KetteiKeisu",
           "HeikinnKetsuatsu",
           "KetsuatsuHendou",
           "MaxKetsuatsu",
           "MinKetsuatsu",
           "PlethysmoHendou",
           "MaxPlethysmo",
           "MinPlethysmo",
           "MinKetsuatsuGosa",
           "MeanKetsuatsuGosa",
           "ZeroPlethyGosa",
           "MaxKetsuatsuGosa",
           "Mu",
           "Eta",
           "Beta_A",
           "Beta",
           "MyHR",
           "Event",
           "KiritsuTime",
           "RR",
           "Stable",
           "result",
           "HR",
           "HF",
           "LF",
           "LF/HF",
           "MeanRR",
           "CVRR"]

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




"""
### フラグ抽出 ##################################################################
# イベント情報の閾値を0.25とする．
# 0.3　<= IVENT < 0.7 : 小フラグ
# 2.0<= IVENT <3.0 : 大フラグ
# にしておけば多分大丈夫．
"""

### 大フラグ抽出
allFlag = df_subA_bio["イベント情報"]
largeFlagRaw= []
for i_raw in range(len(allFlag)):
    if allFlag[i_raw] >= 2.0 and allFlag[i_raw] < 3.0:
        largeFlagRaw.append(i_raw)
#print(largeFlagRaw)

# graph check
#plt.figure(2)
#plt.plot(df_subA_bio["Time"], df_subA_bio["イベント情報"], color = [0, 1, 0])
#plt.plot(df_subA_bio["Time"][largeFlagRaw], df_subA_bio["イベント情報"][largeFlagRaw],\
#         color = [1, 0, 0], linestyle = "None", marker= ".")
#plt.show()


### 小フラグ抽出
allFlag = df_subA_bio["イベント情報"]

smallFlagRaw= []
for i_raw in range(len(allFlag)):
    if allFlag[i_raw] >= 0.3 and allFlag[i_raw] < 0.7:
        smallFlagRaw.append(i_raw)
#print(smallFlagRaw)

#graph check
plt.figure(3)
plt.plot(df_subA_bio["Time"], df_subA_bio["イベント情報"], color = [0, 1, 0])
plt.plot(df_subA_bio["Time"][smallFlagRaw], df_subA_bio["イベント情報"][smallFlagRaw],\
         color = [1, 0, 0], linestyle = "None", marker= ".")
plt.show()

### フラグ番号（刺激前安静:1，刺激:2，アンケート:3，それ以外:0）
# 刺激時フラグを2にする
flag_number = np.zeros(shape=(len(df_subA_bio)))
for i in range(len(largeFlagRaw)):
    flag_number[largeFlagRaw[i]] = 2
# 刺激前安静時フラグを1にする（tmp_numlistは刺激前安静開始時の小フラグ番号を手動設定）
tmp_numlist = [4, 7, 10, 13, 17, 20, 23, 26]
for i in range(len(tmp_numlist)):
    tmp = smallFlagRaw[tmp_numlist[i] - 1]
    while tmp < smallFlagRaw[tmp_numlist[i]]:
        flag_number[tmp] = 1
        tmp = tmp + 1
        if flag_number[tmp] == 2:
            break
# アンケート時フラグを3にする
for i in range(len(tmp_numlist)):
    tmp = smallFlagRaw[tmp_numlist[i]]
    while tmp < smallFlagRaw[tmp_numlist[i] + 1]:
        flag_number[tmp] = 3
        tmp = tmp + 1

df_subA_bio["EventNum"] = flag_number


# 微分します#####################################################################

df_diff = df_subA_bio[["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]]
df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                       columns=["diffμ", "diffη", "diffβ", "diffHR", "diffL/H30", "diffHF30", "diffLF30"])

df_subA_bio2 = pd.concat([df_subA_bio, df_diff], axis=1)


# セッション番号抽出 ##############################################################
session_number = np.zeros(shape=(len(df_subA_bio))) # session1,2がそれぞれ1,2で，それ以外は0
tmp_sessionlist = [[3, 15], [16, 28]]
# session1
tmp = smallFlagRaw[tmp_sessionlist[0][0]]
while tmp <= smallFlagRaw[tmp_sessionlist[0][1]]:
    session_number[tmp] = 1
    tmp = tmp+1
# session2
tmp = smallFlagRaw[tmp_sessionlist[1][0]]
while tmp <= smallFlagRaw[tmp_sessionlist[1][1]]:
    session_number[tmp] = 2
    tmp = tmp+1        

df_subA_bio2["SessionNum"] = session_number



# 変動係数算出 #################################################################

df_cv = df_subA_bio[["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]]
df_cv = pd.DataFrame(sp.variation(df_cv.values, axis=0),\
                     index=["cvμ", "cvη", "cvβ", "cvHR", "cvL/H30", "cvHF30", "cvLF30"])






