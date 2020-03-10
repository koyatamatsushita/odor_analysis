# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:32:35 2019

@author: A_lifePC
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
from scipy import fftpack
import matplotlib.pyplot as plt

# 【前処理】 ####################################################################
### 条件に応じて書き換える
fs = 0.15 # カットオフ周波数
plot_graph = True
bio_drow = True
stan_drow = False
stim_drow = False
lowfilt_drow = False
highfilt_drow = False
diff_drow = False
lowdiff_drow = False
#sub_list = ["Sub_A", "Sub_B", "Sub_C", "Sub_D", "Sub_E", "Sub_F"]
sub_list = ["Sub_A", "Sub_D", "Sub_E", "Sub_F"]
#plot_bio_list = ["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]
#plot_bio_list = ["η", "β", "HR", "L/H30"]
plot_bio_list = ["β"]

### 以下はデータが変わらん限り固定でOK
DATAPATH_LIST = ["C:/Users/A_lifePC/Desktop/ファイル読み込み用/2018_12/快臭",
                 "C:/Users/A_lifePC/Desktop/ファイル読み込み用/2018_12/不快臭",
                 "C:/Users/A_lifePC/Desktop/ファイル読み込み用/2018_12/無臭"]
FILENAME_LIST_BIO = [["subA_ple.csv",
                      "subB_ple.csv",
                      "subC_ple.csv",
                      "subD_ple.csv",
                      "subE_ple.csv",
                      "subF_ple.csv"],
                     ["subA_un.csv",
                      "subB_un.csv",
                      "subC_un.csv",
                      "subD_un.csv",
                      "subE_un.csv",
                      "subF_un.csv"],
                     ["subA_non.csv",
                      "subB_non.csv",
                      "subC_non.csv",
                      "subD_non.csv",
                      "subE_non.csv",
                      "subF_non.csv"]]

## 以下のフラグ番号はグラフを見て調べる必要あり
# セッションフラグ番号
FLAG_NUM_LIST_1 = [[3, 14], [16, 27]]
# 刺激開始フラグ番号
FLAG_NUM_LIST_2 = [4, 7, 10, 13, 17, 20, 23, 26]

# 刺激等の時間(in session)
REST_START = [60, 164, 268, 372] #REST_ENDは+24
STIM_START = [84, 188, 292, 396] #STIM_ENDは+24
QUES_START = [113, 217, 321, 425] # QUES_ENDは+45

SUB_DICT = {"Sub_A":0, "Sub_B":1,"Sub_C":2,"Sub_D":3,"Sub_E":4,"Sub_F":5}

# 【関数定義】 ##################################################################

# 【メイン処理】 ##################################################################

# data読み取り
list_data_p = []
list_data_u = []
list_data_n = []

list_data_p_s1 = []
list_data_u_s1 = []
list_data_n_s1 = []

list_data_p_s2 = []
list_data_u_s2 = []
list_data_n_s2 = []

for i_num, odor in enumerate(["pleasant", "unpleasant", "odorless"]):
    os.chdir(DATAPATH_LIST[i_num])
    for i_sub in range(len(FILENAME_LIST_BIO[i_num])):
        # encoding="cp932"を付けないと文字コードエラー
        df_bio = pd.read_csv(FILENAME_LIST_BIO[i_num][i_sub], encoding="cp932")
        
        # graph check
#        plt.figure()
#        plt.plot(df_bio["Time"], df_bio["β"])
#        plt.show()
        
        ### フラグ抽出 ###
        ## 大フラグ抽出
        allFlag = df_bio["イベント情報"]
        largeFlagRaw= []
        for i_raw in range(len(allFlag)):
            if allFlag[i_raw] >= 2.0 and allFlag[i_raw] < 3.0:
                largeFlagRaw.append(i_raw)
#        print(largeFlagRaw)
        
        # graph check
#        plt.figure()
#        plt.plot(df_bio["Time"], df_bio["イベント情報"], color = [0, 1, 0])
#        plt.plot(df_bio["Time"][largeFlagRaw], df_bio["イベント情報"][largeFlagRaw],\
#                 color = [1, 0, 0], linestyle = "None", marker= ".")
#        plt.show()
        
        
        ## 小フラグ抽出
        allFlag = df_bio["イベント情報"]
        
        smallFlagRaw= []
        for i_raw in range(len(allFlag)):
            if allFlag[i_raw] >= 0.3 and allFlag[i_raw] < 0.7:
                smallFlagRaw.append(i_raw)
#        print(smallFlagRaw)
        
        #graph check
#        plt.figure()
#        plt.plot(df_bio["Time"], df_bio["イベント情報"], color = [0, 1, 0])
#        plt.plot(df_bio["Time"][smallFlagRaw], df_bio["イベント情報"][smallFlagRaw],\
#                 color = [1, 0, 0], linestyle = "None", marker= ".")
#        plt.show()
        
        ## フラグ番号（刺激前安静:1，刺激:2，アンケート:3，それ以外:0）
        # 刺激時フラグを2にする
        flag_number = np.zeros(shape=(len(df_bio)))
        for i in range(len(largeFlagRaw)):
            flag_number[largeFlagRaw[i]] = 2
        # 刺激前安静時フラグを1にする（tmp_numlistは刺激前安静開始時の小フラグ番号を手動設定）
        tmp_numlist = FLAG_NUM_LIST_2
        for i in range(len(tmp_numlist)):
            tmp = smallFlagRaw[tmp_numlist[i] - 1]
            while 1:
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
        
        df_bio["EventNum"] = flag_number
        
        
        
        ### 微分します###
        bio_list = ["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]
        df_diff = df_bio[bio_list]
        df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                               columns=["μ_diff", "η_diff", "β_diff", "HR_diff", "L/H30_diff", "HF30_diff", "LF30_diff"])
        
        df_bio = pd.concat([df_bio, df_diff], axis=1)
        
        
        ### セッション番号抽出 ###
        session_number = np.zeros(shape=(len(df_bio))) # session1,2がそれぞれ1,2で，それ以外は0
        tmp_sessionlist = FLAG_NUM_LIST_1
        # session1
        tmp = smallFlagRaw[tmp_sessionlist[0][0]-1]
        while tmp <= smallFlagRaw[tmp_sessionlist[0][1]]:
            session_number[tmp] = 1
            tmp = tmp+1
        # session2
        tmp = smallFlagRaw[tmp_sessionlist[1][0]-1]
        while tmp <= smallFlagRaw[tmp_sessionlist[1][1]]:
            session_number[tmp] = 2
            tmp = tmp+1        
        
        df_bio["SessionNum"] = session_number
        
        
        ### 標準化 ###
        for bio_name in bio_list:
            tmp = df_bio[bio_name].values
            df_bio[bio_name + "_stan"] = (tmp - np.nanmean(tmp))/np.nanstd(tmp, ddof=1)
        
        
        ### フィルタ処理 ###
        df_bio_fillna = df_bio.fillna(method="ffill") # nanをほかの値に置き換える
        df_bio_fillna = df_bio.fillna(0)
        for bio_name in bio_list:
            tmp = df_bio_fillna[bio_name].values
            tmp = (tmp - np.nanmean(tmp))/np.nanstd(tmp, ddof=1) # 標準化
            n = len(tmp)
            # FFT処理と周波数スケールの作成
            tmpf = fftpack.fft(tmp)/(n/2)
            freq = fftpack.fftfreq(n)
            
            ## フィルタ処理
            # ローパス
            # カットオフ周波数以上に対応するデータを0とする
            tmpf_low = np.copy(tmpf)
            tmpf_low[(freq > fs)] = 0
            tmpf_low[(freq < 0)] = 0
            # 逆FFT処理
            tmp_low = np.real(fftpack.ifft(tmpf_low)*n)
            
            # ハイパス
            # カットオフ周波数以下で対応するデータを0とする
            tmpf_high = np.copy(tmpf)
            tmpf_high[(freq < fs)] = 0
            tmpf_high[(freq < 0)] = 0
            # 逆FFT処理
            tmp_high = np.real(fftpack.ifft(tmpf_high)*n)
            
            df_bio[bio_name + "_low"] = tmp_low
            df_bio[bio_name + "_high"] = tmp_high

        ### ローパスを微分します ###
        bio_list = ["μ_low", "η_low", "β_low", "HR_low", "L/H30_low", "HF30_low", "LF30_low"]
        df_diff = df_bio[bio_list]
        df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                               columns=["μ_low_diff", "η_low_diff", "β_low_diff", "HR_low_diff", "L/H30_low_diff", "HF30_low_diff", "LF30_low_diff"])
        
        df_bio = pd.concat([df_bio, df_diff], axis=1)
        
        


        ### セッション毎にデータを切り分ける ###
        df_bio_s1 = df_bio[df_bio["SessionNum"]==1]
        tmp = df_bio_s1["Time"].iloc[0]
        df_bio_s1 = df_bio_s1.assign(Time = df_bio_s1["Time"].sub(tmp))

        df_bio_s2 = df_bio[df_bio["SessionNum"]==2]
        tmp = df_bio_s2["Time"].iloc[0]
        df_bio_s2 = df_bio_s2.assign(Time = df_bio_s2["Time"].sub(tmp))
        
        
        
        ### 標準化（セッション毎） ###
        for bio_name in bio_list:
            tmp = df_bio_s1[bio_name].values
            df_bio_s1[bio_name + "_stan"] = (tmp - np.nanmean(tmp))/np.nanstd(tmp, ddof=1)
            tmp = df_bio_s2[bio_name].values
            df_bio_s2[bio_name + "_stan"] = (tmp - np.nanmean(tmp))/np.nanstd(tmp, ddof=1)
        
        
        if odor == "pleasant":
            list_data_p.append(df_bio)
            list_data_p_s1.append(df_bio_s1)
            list_data_p_s2.append(df_bio_s2)
        elif odor == "unpleasant":
            list_data_u.append(df_bio)
            list_data_u_s1.append(df_bio_s1)
            list_data_u_s2.append(df_bio_s2)
        elif odor == "odorless":
            list_data_n.append(df_bio)
            list_data_n_s1.append(df_bio_s1)
            list_data_n_s2.append(df_bio_s2)



# 刺激中のみを抽出
list_data_p_s1_stim = []
list_data_u_s1_stim = []
list_data_n_s1_stim = []
for i_sub in range(len(FILENAME_LIST_BIO[0])):
    tmp_p = []
    tmp_u = []
    tmp_n = []
    for i_task in range(len(STIM_START)):
        tmp_p.append(list_data_p_s1[i_sub][(list_data_p_s1[i_sub]["Time"]>STIM_START[i_task]) & (list_data_p_s1[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_u.append(list_data_u_s1[i_sub][(list_data_u_s1[i_sub]["Time"]>STIM_START[i_task]) & (list_data_u_s1[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_n.append(list_data_n_s1[i_sub][(list_data_n_s1[i_sub]["Time"]>STIM_START[i_task]) & (list_data_n_s1[i_sub]["Time"]<STIM_START[i_task]+24)])
    list_data_p_s1_stim.append(tmp_p)
    list_data_u_s1_stim.append(tmp_u)
    list_data_n_s1_stim.append(tmp_n)

list_data_p_s2_stim = []
list_data_u_s2_stim = []
list_data_n_s2_stim = []
for i_sub in range(len(FILENAME_LIST_BIO[0])):
    tmp_p = []
    tmp_u = []
    tmp_n = []
    for i_task in range(len(STIM_START)):
        tmp_p.append(list_data_p_s2[i_sub][(list_data_p_s2[i_sub]["Time"]>STIM_START[i_task]) & (list_data_p_s2[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_u.append(list_data_u_s2[i_sub][(list_data_u_s2[i_sub]["Time"]>STIM_START[i_task]) & (list_data_u_s2[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_n.append(list_data_n_s2[i_sub][(list_data_n_s2[i_sub]["Time"]>STIM_START[i_task]) & (list_data_n_s2[i_sub]["Time"]<STIM_START[i_task]+24)])
    list_data_p_s2_stim.append(tmp_p)
    list_data_u_s2_stim.append(tmp_u)
    list_data_n_s2_stim.append(tmp_n)



# 安静中と刺激中を抽出
list_data_p_s1_rest_stim = []
list_data_u_s1_rest_stim = []
list_data_n_s1_rest_stim = []
for i_sub in range(len(FILENAME_LIST_BIO[0])):
    tmp_p = []
    tmp_u = []
    tmp_n = []
    for i_task in range(len(STIM_START)):
        tmp_p.append(list_data_p_s1[i_sub][(list_data_p_s1[i_sub]["Time"]>REST_START[i_task]) & (list_data_p_s1[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_u.append(list_data_u_s1[i_sub][(list_data_u_s1[i_sub]["Time"]>REST_START[i_task]) & (list_data_u_s1[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_n.append(list_data_n_s1[i_sub][(list_data_n_s1[i_sub]["Time"]>REST_START[i_task]) & (list_data_n_s1[i_sub]["Time"]<STIM_START[i_task]+24)])
    list_data_p_s1_rest_stim.append(tmp_p)
    list_data_u_s1_rest_stim.append(tmp_u)
    list_data_n_s1_rest_stim.append(tmp_n)

list_data_p_s2_rest_stim = []
list_data_u_s2_rest_stim = []
list_data_n_s2_rest_stim = []
for i_sub in range(len(FILENAME_LIST_BIO[0])):
    tmp_p = []
    tmp_u = []
    tmp_n = []
    for i_task in range(len(STIM_START)):
        tmp_p.append(list_data_p_s2[i_sub][(list_data_p_s2[i_sub]["Time"]>REST_START[i_task]) & (list_data_p_s2[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_u.append(list_data_u_s2[i_sub][(list_data_u_s2[i_sub]["Time"]>REST_START[i_task]) & (list_data_u_s2[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_n.append(list_data_n_s2[i_sub][(list_data_n_s2[i_sub]["Time"]>REST_START[i_task]) & (list_data_n_s2[i_sub]["Time"]<STIM_START[i_task]+24)])
    list_data_p_s2_rest_stim.append(tmp_p)
    list_data_u_s2_rest_stim.append(tmp_u)
    list_data_n_s2_rest_stim.append(tmp_n)


## 変動係数
for bio_name in plot_bio_list:
    tmp_p = []
    for i_sub in sub_list:
        tmp_p.append(list_data_p[SUB_DICT[i_sub]][bio_name].values)
    tmp_p = np.array(tmp_p) # エラー


### グラフを描画する ###
if plot_graph == True:
    # セッション毎
    for bio_name in plot_bio_list:
        for i_sub in sub_list:
            fig = plt.figure()
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)
            ax6 = fig.add_subplot(326)
            
            ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
            
            if bio_drow == True:
                ax1.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]][bio_name],\
                         label="pleasant_session1_" + bio_name, color = "darkgreen")
                ax2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]][bio_name],\
                         label="pleasant_session2_" + bio_name, color = "darkgreen")
                ax3.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]][bio_name],\
                         label="unpleasant_session1_" + bio_name, color = "darkgreen")
                ax4.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]][bio_name],\
                         label="unpleasant_session2_" + bio_name, color = "darkgreen")
                ax5.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]][bio_name],\
                         label="odorless_session1_" + bio_name, color = "darkgreen")
                ax6.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]][bio_name],\
                         label="odorless_session2_" + bio_name, color = "darkgreen")
                
                for ax in ax_list:
                    ax.set_ylim(0, 1)
                
    
            
            if stan_drow == True:
                ax1.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]][bio_name + "_stan"],\
                         label="pleasant_session1_" + bio_name, color = "darkgreen")
                ax2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]][bio_name + "_stan"],\
                         label="pleasant_session2_" + bio_name, color = "darkgreen")
                ax3.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]][bio_name + "_stan"],\
                         label="unpleasant_session1_" + bio_name, color = "darkgreen")
                ax4.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]][bio_name + "_stan"],\
                         label="unpleasant_session2_" + bio_name, color = "darkgreen")
                ax5.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]][bio_name + "_stan"],\
                         label="odorless_session1_" + bio_name, color = "darkgreen")
                ax6.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]][bio_name + "_stan"],\
                         label="odorless_session2_" + bio_name, color = "darkgreen")
            
                for ax in ax_list:
                    ax.set_ylim(-1, 1)
            
            # イベント情報込みの場合　以下を記述
            if stim_drow == True:
                ax1_2 = ax1.twinx()
                ax2_2 = ax2.twinx()
                ax3_2 = ax3.twinx()
                ax4_2 = ax4.twinx()
                ax5_2 = ax5.twinx()
                ax6_2 = ax6.twinx()   
                ax1_2.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]]["イベント情報"],\
                           label="pleasant_session1_" + bio_name, color = "orange")
                ax2_2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]]["イベント情報"],\
                           label="pleasant_session2_" + bio_name, color = "orange")
                ax3_2.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]]["イベント情報"],\
                           label="unpleasant_session1_" + bio_name, color = "orange")
                ax4_2.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]]["イベント情報"],\
                           label="unpleasant_session2_" + bio_name, color = "orange")
                ax5_2.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]]["イベント情報"],\
                           label="odorless_session1_" + bio_name, color = "orange")
                ax6_2.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]]["イベント情報"],\
                           label="odorless_session2_" + bio_name, color = "orange")
            
    
    
            if lowfilt_drow == True:
                ax1.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="pleasant_session1_" + bio_name + "_low " + str(fs) + "[Hz]", color = "royalblue")
                ax2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="pleasant_session2_" + bio_name + "_low " + str(fs) + "[Hz]", color = "royalblue")
                ax3.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="unpleasant_session1_" + bio_name + "_low " + str(fs) + "[Hz]", color = "royalblue")
                ax4.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="unpleasant_session2_" + bio_name + "_low " + str(fs) + "[Hz]", color = "royalblue")
                ax5.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="odorless_session1_" + bio_name + "_low " + str(fs) + "[Hz]", color = "royalblue")
                ax6.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="odorless_session2_" + bio_name + "_low " + str(fs) + "[Hz]", color = "royalblue")
                
                for ax in ax_list:
                    ax.set_ylim(-1, 1)

                
            if highfilt_drow == True:
                ax1.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="pleasant_session1_" + bio_name + "_high " + str(fs) + "[Hz]", color = "darkred")
                ax2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="pleasant_session2_" + bio_name + "_high " + str(fs) + "[Hz]", color = "darkred")
                ax3.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="unpleasant_session1_" + bio_name + "_high " + str(fs) + "[Hz]", color = "darkred")
                ax4.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="unpleasant_session2_" + bio_name + "_high " + str(fs) + "[Hz]", color = "darkred")
                ax5.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="odorless_session1_" + bio_name + "_high " + str(fs) + "[Hz]", color = "darkred")
                ax6.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="odorless_session2_" + bio_name + "_high " + str(fs) + "[Hz]", color = "darkred")

                for ax in ax_list:
                    ax.set_ylim(-1, 1)
            
            
            if diff_drow == True:
                ax1.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]][bio_name + "_diff"],\
                           label="pleasant_session1_" + bio_name + "_diff", color = "black")
                ax2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]][bio_name + "_diff"],\
                           label="pleasant_session2_" + bio_name + "_diff", color = "black")
                ax3.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]][bio_name + "_diff"],\
                           label="unpleasant_session1_" + bio_name + "_diff", color = "black")
                ax4.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]][bio_name + "_diff"],\
                           label="unpleasant_session2_" + bio_name + "_diff", color = "black")
                ax5.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]][bio_name + "_diff"],\
                           label="odorless_session1_" + bio_name + "_diff", color = "black")
                ax6.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]][bio_name + "_diff"],\
                           label="odorless_session2_" + bio_name + "_diff", color = "black")

                for ax in ax_list:
                    ax.set_ylim(-1, 1)

    
            
            if lowdiff_drow == True:
                ax1.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]][bio_name + "_low_diff"],\
                           label="pleasant_session1_" + bio_name + "_low_diff", color = "gray")
                ax2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]][bio_name + "_low_diff"],\
                           label="pleasant_session2_" + bio_name + "_low_diff", color = "gray")
                ax3.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]][bio_name + "_low_diff"],\
                           label="unpleasant_session1_" + bio_name + "_low_diff", color = "gray")
                ax4.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]][bio_name + "_low_diff"],\
                           label="unpleasant_session2_" + bio_name + "_low_diff", color = "gray")
                ax5.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]][bio_name + "_low_diff"],\
                           label="odorless_session1_" + bio_name + "_low_diff", color = "gray")
                ax6.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]][bio_name + "_low_diff"],\
                           label="odorless_session2_" + bio_name + "_low_diff", color = "gray")
            
                for ax in ax_list:
                    ax.set_ylim(-1, 1)

            
            
            # 領域を色付け
            for i_num in range(6):
                for i in range(len(REST_START)):
                    # REST
                    ax_list[i_num].axvspan(REST_START[i], REST_START[i]+24, color=(0, 0, 0.9), alpha=0.2)
                    # STIM
                    ax_list[i_num].axvspan(STIM_START[i], STIM_START[i]+24, color=(0.9, 0, 0), alpha=0.2)
                    # QUES
                    ax_list[i_num].axvspan(QUES_START[i], QUES_START[i]+45, color=(0, 0.9, 0), alpha=0.2)
            
            fig.suptitle(i_sub)
            ax1.set_title("pleasant_session1_" + bio_name)
            ax2.set_title("pleasant_session2_" + bio_name)
            ax3.set_title("unpleasant_session1_" + bio_name)
            ax4.set_title("unpleasant_session2_" + bio_name)
            ax5.set_title("odorless_session1_" + bio_name)
            ax6.set_title("odorless_session2_" + bio_name)
            
            plt.tight_layout()
            plt.show()
