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
from scipy import signal
import scipy.interpolate as interp
import matplotlib.pyplot as plt

# 【前処理】 ####################################################################
### 条件に応じて書き換える
data_calculation = True
fc = 0.15 # カットオフ周波数
fs = 1 # サンプリング周波数
plot_graph = True

bio_drow = True
stan_drow = False
stim_drow = False
lowfilt_drow = False
highfilt_drow = False
diff_drow = False
lowdiff_drow = False
highdiff_drow = False

save_plot = False

alldata_plot = False
all_ttest = False

powerspectraｌ = False

#sub_list = ["Sub_A", "Sub_B", "Sub_C", "Sub_D", "Sub_E", "Sub_F"]
sub_list = ["Sub_A", "Sub_D", "Sub_E", "Sub_F"]
#plot_bio_list = ["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]
plot_bio_list = ["η", "β", "τ", "HR", "L/H30"]
#plot_bio_list = ["β"]

### 以下はデータが変わらん限り固定でOK
DATAPATH_LIST = ["C:/Users/A_lifePC/Desktop/ファイル読み込み用/2018_12/快臭",
                 "C:/Users/A_lifePC/Desktop/ファイル読み込み用/2018_12/不快臭",
                 "C:/Users/A_lifePC/Desktop/ファイル読み込み用/2018_12/無臭"]

DATAPATH_SAVE = "C:/Users/A_lifePC/Desktop/ファイル書き出し用/2018_12/plot_graph"

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
if data_calculation == True:
    # data読み取り
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
    
    
            ### セッション毎にデータを切り分ける ###
            df_bio_s1 = df_bio[df_bio["SessionNum"]==1]
            tmp = df_bio_s1["Time"].iloc[0]
            df_bio_s1 = df_bio_s1.assign(Time = df_bio_s1["Time"].sub(tmp))
    
            df_bio_s2 = df_bio[df_bio["SessionNum"]==2]
            tmp = df_bio_s2["Time"].iloc[0]
            df_bio_s2 = df_bio_s2.assign(Time = df_bio_s2["Time"].sub(tmp))
            
            
            ### スプライン補間 ###
            bio_list = ["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]
            df_bio_s1_spline = pd.DataFrame()
            xx = np.arange(0, 471) # 大体1sessionは471[s]
            df_bio_s1_spline["Time"] = xx
            for i_bio in bio_list:
                df_tmp = df_bio_s1.dropna(subset=[i_bio])
                df_bio_s1_spline[i_bio] = interp.spline(df_tmp["Time"], df_tmp[i_bio], xx)
                
            df_bio_s2_spline = pd.DataFrame()
            xx = np.arange(0, 471)
            df_bio_s2_spline["Time"] = xx
            for i_bio in ["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]:
                df_tmp = df_bio_s2.dropna(subset=[i_bio])
                df_bio_s2_spline[i_bio] = interp.spline(df_tmp["Time"], df_tmp[i_bio], xx)
            
            
            ### 補正粘性の算出 ###
            for s in [df_bio_s1_spline, df_bio_s2_spline]:
                s["τ"] = (s["η"] / s["β"]).values
            
            
            ### 微分します###
            df_diff = df_bio_s1_spline[bio_list]
            df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                                   columns=["μ_diff", "η_diff", "β_diff", "HR_diff", "L/H30_diff", "HF30_diff", "LF30_diff"])
            
            df_bio_s1_spline = pd.concat([df_bio_s1_spline, df_diff], axis=1)
    
            df_diff = df_bio_s2_spline[bio_list]
            df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                                   columns=["μ_diff", "η_diff", "β_diff", "HR_diff", "L/H30_diff", "HF30_diff", "LF30_diff"])
            
            df_bio_s2_spline = pd.concat([df_bio_s2_spline, df_diff], axis=1)
            
            
            ### 標準化（セッション毎） ###
            for bio_name in bio_list:
                tmp = df_bio_s1_spline[bio_name].values
                df_bio_s1_spline[bio_name + "_stan"] = sp.zscore(tmp)
                tmp = df_bio_s2_spline[bio_name].values
                df_bio_s2_spline[bio_name + "_stan"] = sp.zscore(tmp)
            
            
            ### フィルタ処理 ###
            for bio_name in bio_list:
                
                ### スムーザ使用 (https://org-technology.com/posts/smoother.html)
                ## session1
                tmp = df_bio_s1_spline[bio_name].values
                n = len(tmp)
                # FFT処理と周波数スケールの作成
                tmpf = fftpack.fft(tmp)/(n/2)
                tmpf[0] = tmpf[0]/2 # 直流成分の振幅を揃える(実用上は不要)
                freq = fftpack.fftfreq(n)
                
                ## フィルタ処理
                
                # ローパス
                # カットオフ周波数以上に対応するデータを0とする
                tmpf_low = np.copy(tmpf)
                tmpf_low[(freq > fc)] = 0
                tmpf_low[(freq < 0)] = 0
                # 逆FFT処理
                tmp_low = np.real(fftpack.ifft(tmpf_low)*n)
                
                # ハイパス
                # カットオフ周波数以下で対応するデータを0とする
                tmpf_high = np.copy(tmpf)
                tmpf_high[(freq < fc)] = 0
                tmpf_high[(freq < 0)] = 0
                # 逆FFT処理
                tmp_high = np.real(fftpack.ifft(tmpf_high)*n)
                
                df_bio_s1_spline[bio_name + "_low"] = tmp_low
                df_bio_s1_spline[bio_name + "_high"] = tmp_high
                
                ## session2
                tmp = df_bio_s2_spline[bio_name].values
                n = len(tmp)
                # FFT処理と周波数スケールの作成
                tmpf = fftpack.fft(tmp)/(n/2)
                tmpf[0] = tmpf[0]/2 # 直流成分の振幅を揃える(実用上は不要)
                freq = fftpack.fftfreq(n)
                
                ## フィルタ処理
                # ローパス
                # カットオフ周波数以上に対応するデータを0とする
                tmpf_low = np.copy(tmpf)
                tmpf_low[(freq > fc)] = 0
                tmpf_low[(freq < 0)] = 0
                # 逆FFT処理
                tmp_low = np.real(fftpack.ifft(tmpf_low)*n)
                
                # ハイパス
                # カットオフ周波数以下で対応するデータを0とする
                tmpf_high = np.copy(tmpf)
                tmpf_high[(freq < fc)] = 0
                tmpf_high[(freq < 0)] = 0
                # 逆FFT処理
                tmp_high = np.real(fftpack.ifft(tmpf_high)*n)
                
                df_bio_s2_spline[bio_name + "_low"] = tmp_low
                df_bio_s2_spline[bio_name + "_high"] = tmp_high
                
                """
                ### numpyで除去　(https://qiita.com/ajiron/items/ca630de8b6e3ed28ad1e)
                ##session1
                tmp = df_bio_s1_spline[bio_name].values
                n = len(tmp)
                freq = np.linspace(0, fs, n)
                tmpf = np.fft.fft(tmp)
                tmpf_low = tmpf.copy()
                tmpf_low[freq > fc] = 0 + 0j
                tmp_low = np.fft.ifft(tmpf_low)
                tmp_low = tmp_low.real
                
                tmpf_high = tmpf.copy()
                tmpf_high[freq < fc] = 0 + 0j
                tmp_high = np.fft.ifft(tmpf_high)
                tmp_high = tmp_high.real
                
                df_bio_s1_spline[bio_name + "_low"] = tmp_low
                df_bio_s1_spline[bio_name + "_high"] = tmp_high
                
                ##session2
                tmp = df_bio_s2_spline[bio_name].values
                n = len(tmp)
                freq = np.linspace(0, fs, n)
                tmpf = np.fft.fft(tmp)
                tmpf_low = tmpf.copy()
                tmpf_low[freq > fc] = 0 + 0j
                tmp_low = np.fft.ifft(tmpf_low)
                tmp_low = tmp_low.real
                
                tmpf_high = tmpf.copy()
                tmpf_high[freq < fc] = 0 + 0j
                tmp_high = np.fft.ifft(tmpf_high)
                tmp_high = tmp_high.real
                
                df_bio_s2_spline[bio_name + "_low"] = tmp_low
                df_bio_s2_spline[bio_name + "_high"] = tmp_high
                """
    
            ### ローパスを微分します ###
            bio_list = ["μ_low", "η_low", "β_low", "HR_low", "L/H30_low", "HF30_low", "LF30_low"]
            df_diff = df_bio_s1_spline[bio_list]
            df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                                   columns=["μ_low_diff", "η_low_diff", "β_low_diff", "HR_low_diff", "L/H30_low_diff", "HF30_low_diff", "LF30_low_diff"])
            df_bio_s1_spline = pd.concat([df_bio_s1_spline, df_diff], axis=1)
    
            bio_list = ["μ_low", "η_low", "β_low", "HR_low", "L/H30_low", "HF30_low", "LF30_low"]
            df_diff = df_bio_s2_spline[bio_list]
            df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                                   columns=["μ_low_diff", "η_low_diff", "β_low_diff", "HR_low_diff", "L/H30_low_diff", "HF30_low_diff", "LF30_low_diff"])
            df_bio_s2_spline = pd.concat([df_bio_s2_spline, df_diff], axis=1)
    
            ### ハイパスを微分します ###
            bio_list = ["μ_high", "η_high", "β_high", "HR_high", "L/H30_high", "HF30_high", "LF30_high"]
            df_diff = df_bio_s1_spline[bio_list]
            df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                                   columns=["μ_high_diff", "η_high_diff", "β_high_diff", "HR_high_diff", "L/H30_high_diff", "HF30_high_diff", "LF30_high_diff"])
            df_bio_s1_spline = pd.concat([df_bio_s1_spline, df_diff], axis=1)
    
            bio_list = ["μ_high", "η_high", "β_high", "HR_high", "L/H30_high", "HF30_high", "LF30_high"]
            df_diff = df_bio_s2_spline[bio_list]
            df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                                   columns=["μ_high_diff", "η_high_diff", "β_high_diff", "HR_high_diff", "L/H30_high_diff", "HF30_high_diff", "LF30_high_diff"])
            df_bio_s2_spline = pd.concat([df_bio_s2_spline, df_diff], axis=1)
    
    
            
            if odor == "pleasant":
                list_data_p_s1.append(df_bio_s1_spline)
                list_data_p_s2.append(df_bio_s2_spline)
            elif odor == "unpleasant":
                list_data_u_s1.append(df_bio_s1_spline)
                list_data_u_s2.append(df_bio_s2_spline)
            elif odor == "odorless":
                list_data_n_s1.append(df_bio_s1_spline)
                list_data_n_s2.append(df_bio_s2_spline)



if powerspectraｌ == True:
    bio_list = ["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]
    for bio_name in bio_list:
        
        # パワースペクトル密度計算
        
        #########
        # 工事中 #
        #########
        
        
        # graph plot
        fig = plt.figure(figsize=(18, 9)) 
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.plot(freq, p)
        ax2.plot(freq, p_low)
        ax3.plot(freq, p_high)
        ymax = max([ax1.axis()[3], ax2.axis()[3], ax3.axis()[3]])
        ymin = min([ax1.axis()[2], ax2.axis()[2], ax3.axis()[2]])
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, 0.5)
            ax.set_ylim(ymin, ymax) 
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel(bio_name+"_Amplitude")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel(bio_name+"_low"+"_Amplitude")
        ax3.set_xlabel("Frequency [Hz]")
        ax3.set_ylabel(bio_name+"_high"+"_Amplitude")
        plt.tight_layout()
        plt.show()



### グラフを描画する ###
if plot_graph == True:
    
    # セッション毎
    for bio_name in plot_bio_list:
        for i_sub in sub_list:
            tytle = ""
            fig = plt.figure(figsize=(18,9))
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
                        
                tytle = tytle + "_stan"
            
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
                
                tytle = tytle + "_event"
    
    
            if lowfilt_drow == True:
                ax1.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="pleasant_session1_" + bio_name + "_low " + str(fc) + "[Hz]", color = "royalblue")
                ax2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="pleasant_session2_" + bio_name + "_low " + str(fc) + "[Hz]", color = "royalblue")
                ax3.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="unpleasant_session1_" + bio_name + "_low " + str(fc) + "[Hz]", color = "royalblue")
                ax4.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="unpleasant_session2_" + bio_name + "_low " + str(fc) + "[Hz]", color = "royalblue")
                ax5.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="odorless_session1_" + bio_name + "_low " + str(fc) + "[Hz]", color = "royalblue")
                ax6.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]][bio_name + "_low"],\
                           label="odorless_session2_" + bio_name + "_low " + str(fc) + "[Hz]", color = "royalblue")
                
                tytle = tytle + "_low"
                
            if highfilt_drow == True:
                ax1.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="pleasant_session1_" + bio_name + "_high " + str(fc) + "[Hz]", color = "darkred")
                ax2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="pleasant_session2_" + bio_name + "_high " + str(fc) + "[Hz]", color = "darkred")
                ax3.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="unpleasant_session1_" + bio_name + "_high " + str(fc) + "[Hz]", color = "darkred")
                ax4.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="unpleasant_session2_" + bio_name + "_high " + str(fc) + "[Hz]", color = "darkred")
                ax5.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="odorless_session1_" + bio_name + "_high " + str(fc) + "[Hz]", color = "darkred")
                ax6.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]][bio_name + "_high"],\
                           label="odorless_session2_" + bio_name + "_high " + str(fc) + "[Hz]", color = "darkred")

                tytle = tytle + "_high"
            
            
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
                    
                tytle = tytle + "_diff"

    
            
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
            
                tytle = tytle + "_lowdiff"
                

            if highdiff_drow == True:
                ax1.plot(list_data_p_s1[SUB_DICT[i_sub]]["Time"], list_data_p_s1[SUB_DICT[i_sub]][bio_name + "_high_diff"],\
                           label="pleasant_session1_" + bio_name + "_high_diff", color = "gray")
                ax2.plot(list_data_p_s2[SUB_DICT[i_sub]]["Time"], list_data_p_s2[SUB_DICT[i_sub]][bio_name + "_high_diff"],\
                           label="pleasant_session2_" + bio_name + "_high_diff", color = "gray")
                ax3.plot(list_data_u_s1[SUB_DICT[i_sub]]["Time"], list_data_u_s1[SUB_DICT[i_sub]][bio_name + "_high_diff"],\
                           label="unpleasant_session1_" + bio_name + "_high_diff", color = "gray")
                ax4.plot(list_data_u_s2[SUB_DICT[i_sub]]["Time"], list_data_u_s2[SUB_DICT[i_sub]][bio_name + "_high_diff"],\
                           label="unpleasant_session2_" + bio_name + "_high_diff", color = "gray")
                ax5.plot(list_data_n_s1[SUB_DICT[i_sub]]["Time"], list_data_n_s1[SUB_DICT[i_sub]][bio_name + "_high_diff"],\
                           label="odorless_session1_" + bio_name + "_high_diff", color = "gray")
                ax6.plot(list_data_n_s2[SUB_DICT[i_sub]]["Time"], list_data_n_s2[SUB_DICT[i_sub]][bio_name + "_high_diff"],\
                           label="odorless_session2_" + bio_name + "_high_diff", color = "gray")

                tytle = tytle + "_highdiff"

            ymax = max([ax1.axis()[3], ax2.axis()[3], ax3.axis()[3], ax4.axis()[3], ax5.axis()[3], ax6.axis()[3]])
            ymin = min([ax1.axis()[2], ax2.axis()[2], ax3.axis()[2], ax4.axis()[2], ax5.axis()[2], ax6.axis()[2]])
            for ax in ax_list:
                ax.set_xlim(0, 470)
                ax.set_ylim(ymin, ymax) 
            
            # 領域を色付け
            for i_num in range(len(ax_list)):
                for i in range(len(REST_START)):
                    # REST
                    ax_list[i_num].axvspan(REST_START[i], REST_START[i]+24, color=(0, 0, 0.9), alpha=0.2)
                    # STIM
                    ax_list[i_num].axvspan(STIM_START[i], STIM_START[i]+24, color=(0.9, 0, 0), alpha=0.2)
                    # QUES
                    ax_list[i_num].axvspan(QUES_START[i], QUES_START[i]+45, color=(0, 0.9, 0), alpha=0.2)
            
            fig.suptitle(i_sub, fontsize=20)
            ax1.set_title("pleasant_session1_" + bio_name + tytle, fontsize=10)
            ax2.set_title("pleasant_session2_" + bio_name + tytle, fontsize=10)
            ax3.set_title("unpleasant_session1_" + bio_name + tytle, fontsize=10)
            ax4.set_title("unpleasant_session2_" + bio_name + tytle, fontsize=10)
            ax5.set_title("odorless_session1_" + bio_name + tytle, fontsize=10)
            ax6.set_title("odorless_session2_" + bio_name + tytle, fontsize=10)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
            
            #save plot
            bio_name_ = bio_name.replace("/", "") # スラッシュがある場合には削除
            if save_plot == True:
                if os.path.isdir(DATAPATH_SAVE + "/" + bio_name_ + tytle) == False:
                    os.mkdir(DATAPATH_SAVE + "/" + bio_name_ + tytle)
                os.chdir(DATAPATH_SAVE + "/" + bio_name_ + tytle)
                plt.savefig(i_sub + "_" + bio_name_ + tytle + ".png")



# 刺激中のみを抽出
list_data_p_s1_stim = []
list_data_u_s1_stim = []
list_data_n_s1_stim = []
for i_sub in range(len(FILENAME_LIST_BIO[0])):
    tmp_p = []
    tmp_u = []
    tmp_n = []
    for i_task in range(len(STIM_START)):
        tmp_p.append(list_data_p_s1[i_sub][(list_data_p_s1[i_sub]["Time"]>=STIM_START[i_task]) & (list_data_p_s1[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_u.append(list_data_u_s1[i_sub][(list_data_u_s1[i_sub]["Time"]>=STIM_START[i_task]) & (list_data_u_s1[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_n.append(list_data_n_s1[i_sub][(list_data_n_s1[i_sub]["Time"]>=STIM_START[i_task]) & (list_data_n_s1[i_sub]["Time"]<STIM_START[i_task]+24)])
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
        tmp_p.append(list_data_p_s2[i_sub][(list_data_p_s2[i_sub]["Time"]>=STIM_START[i_task]) & (list_data_p_s2[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_u.append(list_data_u_s2[i_sub][(list_data_u_s2[i_sub]["Time"]>=STIM_START[i_task]) & (list_data_u_s2[i_sub]["Time"]<STIM_START[i_task]+24)])
        tmp_n.append(list_data_n_s2[i_sub][(list_data_n_s2[i_sub]["Time"]>=STIM_START[i_task]) & (list_data_n_s2[i_sub]["Time"]<STIM_START[i_task]+24)])
    list_data_p_s2_stim.append(tmp_p)
    list_data_u_s2_stim.append(tmp_u)
    list_data_n_s2_stim.append(tmp_n)



# 安静中を抽出
list_data_p_s1_rest = []
list_data_u_s1_rest = []
list_data_n_s1_rest = []
for i_sub in range(len(FILENAME_LIST_BIO[0])):
    tmp_p = []
    tmp_u = []
    tmp_n = []
    for i_task in range(len(STIM_START)):
        tmp_p.append(list_data_p_s1[i_sub][(list_data_p_s1[i_sub]["Time"]>=REST_START[i_task]) & (list_data_p_s1[i_sub]["Time"]<REST_START[i_task]+24)])
        tmp_u.append(list_data_u_s1[i_sub][(list_data_u_s1[i_sub]["Time"]>=REST_START[i_task]) & (list_data_u_s1[i_sub]["Time"]<REST_START[i_task]+24)])
        tmp_n.append(list_data_n_s1[i_sub][(list_data_n_s1[i_sub]["Time"]>=REST_START[i_task]) & (list_data_n_s1[i_sub]["Time"]<REST_START[i_task]+24)])
    list_data_p_s1_rest.append(tmp_p)
    list_data_u_s1_rest.append(tmp_u)
    list_data_n_s1_rest.append(tmp_n)

list_data_p_s2_rest = []
list_data_u_s2_rest = []
list_data_n_s2_rest = []
for i_sub in range(len(FILENAME_LIST_BIO[0])):
    tmp_p = []
    tmp_u = []
    tmp_n = []
    for i_task in range(len(STIM_START)):
        tmp_p.append(list_data_p_s2[i_sub][(list_data_p_s2[i_sub]["Time"]>=REST_START[i_task]) & (list_data_p_s2[i_sub]["Time"]<REST_START[i_task]+24)])
        tmp_u.append(list_data_u_s2[i_sub][(list_data_u_s2[i_sub]["Time"]>=REST_START[i_task]) & (list_data_u_s2[i_sub]["Time"]<REST_START[i_task]+24)])
        tmp_n.append(list_data_n_s2[i_sub][(list_data_n_s2[i_sub]["Time"]>=REST_START[i_task]) & (list_data_n_s2[i_sub]["Time"]<REST_START[i_task]+24)])
    list_data_p_s2_rest.append(tmp_p)
    list_data_u_s2_rest.append(tmp_u)
    list_data_n_s2_rest.append(tmp_n)





### 全被験者まとめて計算 ##########################################################
bio_list = ["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]
df_all_bio_s1_spline_p = df_bio_s1_spline["Time"]
df_all_bio_s1_spline_u = df_bio_s1_spline["Time"]
df_all_bio_s1_spline_n = df_bio_s1_spline["Time"]
for bio_name in bio_list:
    df_bio_p = pd.DataFrame([])
    df_bio_u = pd.DataFrame([])
    df_bio_n = pd.DataFrame([])
    for sub_name in sub_list:
        df_bio_p[sub_name] = list_data_p_s1[SUB_DICT[sub_name]][bio_name + "_stan"].values
        df_bio_u[sub_name] = list_data_u_s1[SUB_DICT[sub_name]][bio_name + "_stan"].values
        df_bio_n[sub_name] = list_data_n_s1[SUB_DICT[sub_name]][bio_name + "_stan"].values
    df_mean_p = pd.DataFrame(np.mean(df_bio_p.values, axis=1).T, columns=[bio_name])
    df_mean_u = pd.DataFrame(np.mean(df_bio_u.values, axis=1).T, columns=[bio_name])
    df_mean_n = pd.DataFrame(np.mean(df_bio_n.values, axis=1).T, columns=[bio_name])
    df_std_p = pd.DataFrame(np.std(df_bio_p.values, axis=1, ddof=1).T, columns=[bio_name + "_std"])
    df_std_u = pd.DataFrame(np.std(df_bio_u.values, axis=1, ddof=1).T, columns=[bio_name + "_std"])
    df_std_n = pd.DataFrame(np.std(df_bio_n.values, axis=1, ddof=1).T, columns=[bio_name + "_std"])
    df_cv_p = pd.DataFrame(df_std_p.values / np.mean(df_std_p.values), columns=[bio_name + "_cv"])
    df_cv_u = pd.DataFrame(df_std_u.values / np.mean(df_std_u.values), columns=[bio_name + "_cv"])
    df_cv_n = pd.DataFrame(df_std_n.values / np.mean(df_std_n.values), columns=[bio_name + "_cv"])
    df_all_bio_s1_spline_p = pd.concat([df_all_bio_s1_spline_p, df_mean_p, df_std_p, df_cv_p], axis=1, sort=False)
    df_all_bio_s1_spline_u = pd.concat([df_all_bio_s1_spline_u, df_mean_u, df_std_u, df_cv_u], axis=1, sort=False)
    df_all_bio_s1_spline_n = pd.concat([df_all_bio_s1_spline_n, df_mean_n, df_std_n, df_cv_n], axis=1, sort=False)
    

df_all_bio_s2_spline_p = df_bio_s2_spline["Time"]
df_all_bio_s2_spline_u = df_bio_s2_spline["Time"]
df_all_bio_s2_spline_n = df_bio_s2_spline["Time"]
for bio_name in bio_list:
    df_bio_p = pd.DataFrame([])
    df_bio_u = pd.DataFrame([])
    df_bio_n = pd.DataFrame([])
    for sub_name in sub_list:
        df_bio_p[sub_name] = list_data_p_s2[SUB_DICT[sub_name]][bio_name + "_stan"].values
        df_bio_u[sub_name] = list_data_u_s2[SUB_DICT[sub_name]][bio_name + "_stan"].values
        df_bio_n[sub_name] = list_data_n_s2[SUB_DICT[sub_name]][bio_name + "_stan"].values
    df_mean_p = pd.DataFrame(np.mean(df_bio_p.values, axis=1).T, columns=[bio_name])
    df_mean_u = pd.DataFrame(np.mean(df_bio_u.values, axis=1).T, columns=[bio_name])
    df_mean_n = pd.DataFrame(np.mean(df_bio_n.values, axis=1).T, columns=[bio_name])
    df_std_p = pd.DataFrame(np.std(df_bio_p.values, axis=1, ddof=1).T, columns=[bio_name + "_std"])
    df_std_u = pd.DataFrame(np.std(df_bio_u.values, axis=1, ddof=1).T, columns=[bio_name + "_std"])
    df_std_n = pd.DataFrame(np.std(df_bio_n.values, axis=1, ddof=1).T, columns=[bio_name + "_std"])
    df_cv_p = pd.DataFrame(df_std_p.values / np.mean(df_std_p.values), columns=[bio_name + "_cv"])
    df_cv_u = pd.DataFrame(df_std_u.values / np.mean(df_std_u.values), columns=[bio_name + "_cv"])
    df_cv_n = pd.DataFrame(df_std_n.values / np.mean(df_std_n.values), columns=[bio_name + "_cv"])
    df_all_bio_s2_spline_p = pd.concat([df_all_bio_s2_spline_p, df_mean_p, df_std_p, df_cv_p], axis=1, sort=False)
    df_all_bio_s2_spline_u = pd.concat([df_all_bio_s2_spline_u, df_mean_u, df_std_u, df_cv_u], axis=1, sort=False)
    df_all_bio_s2_spline_n = pd.concat([df_all_bio_s2_spline_n, df_mean_n, df_std_n, df_cv_n], axis=1, sort=False)



# 刺激中のみを抽出
list_all_data_p_s1_stim = []
list_all_data_u_s1_stim = []
list_all_data_n_s1_stim = []
tmp_p = []
tmp_u = []
tmp_n = []
for i_task in range(len(STIM_START)):
    tmp_p.append(df_all_bio_s1_spline_p[(df_all_bio_s1_spline_p["Time"]>=STIM_START[i_task]) & (df_all_bio_s1_spline_p["Time"]<STIM_START[i_task]+24)])
    tmp_u.append(df_all_bio_s1_spline_u[(df_all_bio_s1_spline_u["Time"]>=STIM_START[i_task]) & (df_all_bio_s1_spline_u["Time"]<STIM_START[i_task]+24)])
    tmp_n.append(df_all_bio_s1_spline_n[(df_all_bio_s1_spline_n["Time"]>=STIM_START[i_task]) & (df_all_bio_s1_spline_n["Time"]<STIM_START[i_task]+24)])
list_all_data_p_s1_stim.append(tmp_p)
list_all_data_u_s1_stim.append(tmp_u)
list_all_data_n_s1_stim.append(tmp_n)

list_all_data_p_s2_stim = []
list_all_data_u_s2_stim = []
list_all_data_n_s2_stim = []
tmp_p = []
tmp_u = []
tmp_n = []
for i_task in range(len(STIM_START)):
    tmp_p.append(df_all_bio_s2_spline_p[(df_all_bio_s2_spline_p["Time"]>=STIM_START[i_task]) & (df_all_bio_s2_spline_p["Time"]<STIM_START[i_task]+24)])
    tmp_u.append(df_all_bio_s2_spline_u[(df_all_bio_s2_spline_u["Time"]>=STIM_START[i_task]) & (df_all_bio_s2_spline_u["Time"]<STIM_START[i_task]+24)])
    tmp_n.append(df_all_bio_s2_spline_n[(df_all_bio_s2_spline_n["Time"]>=STIM_START[i_task]) & (df_all_bio_s2_spline_n["Time"]<STIM_START[i_task]+24)])
list_all_data_p_s2_stim.append(tmp_p)
list_all_data_u_s2_stim.append(tmp_u)
list_all_data_n_s2_stim.append(tmp_n)


# 刺激前安静中のみを抽出
list_all_data_p_s1_rest = []
list_all_data_u_s1_rest = []
list_all_data_n_s1_rest = []
tmp_p = []
tmp_u = []
tmp_n = []
for i_task in range(len(STIM_START)):
    tmp_p.append(df_all_bio_s1_spline_p[(df_all_bio_s1_spline_p["Time"]>=REST_START[i_task]) & (df_all_bio_s1_spline_p["Time"]<REST_START[i_task]+24)])
    tmp_u.append(df_all_bio_s1_spline_u[(df_all_bio_s1_spline_u["Time"]>=REST_START[i_task]) & (df_all_bio_s1_spline_u["Time"]<REST_START[i_task]+24)])
    tmp_n.append(df_all_bio_s1_spline_n[(df_all_bio_s1_spline_n["Time"]>=REST_START[i_task]) & (df_all_bio_s1_spline_n["Time"]<REST_START[i_task]+24)])
list_all_data_p_s1_rest.append(tmp_p)
list_all_data_u_s1_rest.append(tmp_u)
list_all_data_n_s1_rest.append(tmp_n)

list_all_data_p_s2_rest = []
list_all_data_u_s2_rest = []
list_all_data_n_s2_rest = []
tmp_p = []
tmp_u = []
tmp_n = []
for i_task in range(len(STIM_START)):
    tmp_p.append(df_all_bio_s2_spline_p[(df_all_bio_s2_spline_p["Time"]>=REST_START[i_task]) & (df_all_bio_s2_spline_p["Time"]<REST_START[i_task]+24)])
    tmp_u.append(df_all_bio_s2_spline_u[(df_all_bio_s2_spline_u["Time"]>=REST_START[i_task]) & (df_all_bio_s2_spline_u["Time"]<REST_START[i_task]+24)])
    tmp_n.append(df_all_bio_s2_spline_n[(df_all_bio_s2_spline_n["Time"]>=REST_START[i_task]) & (df_all_bio_s2_spline_n["Time"]<REST_START[i_task]+24)])
list_all_data_p_s2_rest.append(tmp_p)
list_all_data_u_s2_rest.append(tmp_u)
list_all_data_n_s2_rest.append(tmp_n)
    


# 刺激中の最大，最小，平均，標準偏差
df_stim_s1_p = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_p_s1_stim[0])):
        tmp_max = np.max(list_all_data_p_s1_stim[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_p_s1_stim[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_p_s1_stim[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_p_s1_stim[0][i_task][bio_name].values)
        df["Task"+str(i_task+1)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_stim_s1_p = pd.concat([df_stim_s1_p, df.T], axis=1, sort=False)

df_stim_s2_p = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_p_s2_stim[0])):
        tmp_max = np.max(list_all_data_p_s2_stim[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_p_s2_stim[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_p_s2_stim[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_p_s2_stim[0][i_task][bio_name].values)
        df["Task"+str(i_task+5)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_stim_s2_p = pd.concat([df_stim_s2_p, df.T], axis=1, sort=False)

df_stim_p = pd.concat([df_stim_s1_p, df_stim_s2_p], axis=0, sort=False)


df_stim_s1_u = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_u_s1_stim[0])):
        tmp_max = np.max(list_all_data_u_s1_stim[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_u_s1_stim[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_u_s1_stim[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_u_s1_stim[0][i_task][bio_name].values)
        df["Task"+str(i_task+1)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_stim_s1_u = pd.concat([df_stim_s1_u, df.T], axis=1, sort=False)

df_stim_s2_u = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_u_s2_stim[0])):
        tmp_max = np.max(list_all_data_u_s2_stim[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_u_s2_stim[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_u_s2_stim[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_u_s2_stim[0][i_task][bio_name].values)
        df["Task"+str(i_task+5)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_stim_s2_u = pd.concat([df_stim_s2_u, df.T], axis=1, sort=False)

df_stim_u = pd.concat([df_stim_s1_u, df_stim_s2_u], axis=0, sort=False)


df_stim_s1_n = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_n_s1_stim[0])):
        tmp_max = np.max(list_all_data_n_s1_stim[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_n_s1_stim[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_n_s1_stim[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_n_s1_stim[0][i_task][bio_name].values)
        df["Task"+str(i_task+1)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_stim_s1_n = pd.concat([df_stim_s1_n, df.T], axis=1, sort=False)

df_stim_s2_n = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_n_s2_stim[0])):
        tmp_max = np.max(list_all_data_n_s2_stim[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_n_s2_stim[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_n_s2_stim[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_n_s2_stim[0][i_task][bio_name].values)
        df["Task"+str(i_task+5)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_stim_s2_n = pd.concat([df_stim_s2_n, df.T], axis=1, sort=False)

df_stim_n = pd.concat([df_stim_s1_n, df_stim_s2_n], axis=0, sort=False)


# 刺激前安静中の最大，最小，平均，標準偏差
df_rest_s1_p = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_p_s1_rest[0])):
        tmp_max = np.max(list_all_data_p_s1_rest[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_p_s1_rest[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_p_s1_rest[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_p_s1_rest[0][i_task][bio_name].values)
        df["Task"+str(i_task+1)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_rest_s1_p = pd.concat([df_rest_s1_p, df.T], axis=1, sort=False)

df_rest_s2_p = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_p_s2_rest[0])):
        tmp_max = np.max(list_all_data_p_s2_rest[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_p_s2_rest[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_p_s2_rest[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_p_s2_rest[0][i_task][bio_name].values)
        df["Task"+str(i_task+5)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_rest_s2_p = pd.concat([df_rest_s2_p, df.T], axis=1, sort=False)

df_rest_p = pd.concat([df_rest_s1_p, df_rest_s2_p], axis=0, sort=False)


df_rest_s1_u = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_u_s1_rest[0])):
        tmp_max = np.max(list_all_data_u_s1_rest[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_u_s1_rest[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_u_s1_rest[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_u_s1_rest[0][i_task][bio_name].values)
        df["Task"+str(i_task+1)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_rest_s1_u = pd.concat([df_rest_s1_u, df.T], axis=1, sort=False)

df_rest_s2_u = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_u_s2_rest[0])):
        tmp_max = np.max(list_all_data_u_s2_rest[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_u_s2_rest[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_u_s2_rest[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_u_s2_rest[0][i_task][bio_name].values)
        df["Task"+str(i_task+5)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_rest_s2_u = pd.concat([df_rest_s2_u, df.T], axis=1, sort=False)

df_rest_u = pd.concat([df_rest_s1_u, df_rest_s2_u], axis=0, sort=False)


df_rest_s1_n = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_n_s1_rest[0])):
        tmp_max = np.max(list_all_data_n_s1_rest[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_n_s1_rest[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_n_s1_rest[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_n_s1_rest[0][i_task][bio_name].values)
        df["Task"+str(i_task+1)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_rest_s1_n = pd.concat([df_rest_s1_n, df.T], axis=1, sort=False)

df_rest_s2_n = pd.DataFrame()
for bio_name in bio_list:
    df = pd.DataFrame()
    for i_task in range(len(list_all_data_n_s2_rest[0])):
        tmp_max = np.max(list_all_data_n_s2_rest[0][i_task][bio_name].values)
        tmp_min = np.min(list_all_data_n_s2_rest[0][i_task][bio_name].values)
        tmp_mean = np.mean(list_all_data_n_s2_rest[0][i_task][bio_name].values)
        tmp_std = np.std(list_all_data_n_s2_rest[0][i_task][bio_name].values)
        df["Task"+str(i_task+5)] = tmp_max, tmp_min, tmp_mean, tmp_std
    df.index = [bio_name+"max", bio_name+"min", bio_name+"mean", bio_name+"std"]
    df_rest_s2_n = pd.concat([df_rest_s2_n, df.T], axis=1, sort=False)

df_rest_n = pd.concat([df_rest_s1_n, df_rest_s2_n], axis=0, sort=False)


# 刺激中と刺激前の有意差検定
if all_ttest == True:
    for bio_name in bio_list:
        print(bio_name)
        print(sp.ttest_ind(df_stim_u[bio_name+"max"].values, df_rest_u[bio_name+"max"].values, equal_var=False))
        print(sp.ttest_ind(df_stim_u[bio_name+"min"].values, df_rest_u[bio_name+"min"].values, equal_var=False))
        print(sp.ttest_ind(df_stim_u[bio_name+"mean"].values, df_rest_u[bio_name+"mean"].values, equal_var=False))
        print(sp.ttest_ind(df_stim_u[bio_name+"std"].values, df_rest_u[bio_name+"std"].values, equal_var=False))
        print("")
        print(sp.ttest_ind(df_stim_n[bio_name+"max"].values, df_rest_n[bio_name+"max"].values, equal_var=False))
        print(sp.ttest_ind(df_stim_n[bio_name+"min"].values, df_rest_n[bio_name+"min"].values, equal_var=False))
        print(sp.ttest_ind(df_stim_n[bio_name+"mean"].values, df_rest_n[bio_name+"mean"].values, equal_var=False))
        print(sp.ttest_ind(df_stim_n[bio_name+"std"].values, df_rest_n[bio_name+"std"].values, equal_var=False))
        print("\n")



# 全データまとめをプロットしてみる
if alldata_plot == True:
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.plot(df_all_bio_s1_spline_p["Time"], df_all_bio_s1_spline_p["β"], color = "darkgreen")
    ax2.plot(df_all_bio_s1_spline_u["Time"], df_all_bio_s1_spline_u["β"], color = "darkgreen")
    ax3.plot(df_all_bio_s1_spline_n["Time"], df_all_bio_s1_spline_n["β"], color = "darkgreen")
    std_p = df_all_bio_s1_spline_p["β_std"]
    std_u = df_all_bio_s1_spline_u["β_std"]
    std_n = df_all_bio_s1_spline_n["β_std"]
    ax1.plot(df_all_bio_s1_spline_p["Time"], df_all_bio_s1_spline_p["β"]+std_p, color = "gray")
    ax1.plot(df_all_bio_s1_spline_p["Time"], df_all_bio_s1_spline_p["β"]-std_p, color = "gray")
    ax2.plot(df_all_bio_s1_spline_u["Time"], df_all_bio_s1_spline_u["β"]+std_u, color = "gray")
    ax2.plot(df_all_bio_s1_spline_u["Time"], df_all_bio_s1_spline_u["β"]-std_u, color = "gray")
    ax3.plot(df_all_bio_s1_spline_n["Time"], df_all_bio_s1_spline_n["β"]+std_n, color = "gray")
    ax3.plot(df_all_bio_s1_spline_n["Time"], df_all_bio_s1_spline_n["β"]-std_n, color = "gray")
    ax1.fill_between(df_all_bio_s1_spline_p["Time"], df_all_bio_s1_spline_p["β"]+std_p, df_all_bio_s1_spline_p["β"]-std_p,\
                     facecolor="gray", alpha=0.8)
    ax2.fill_between(df_all_bio_s1_spline_u["Time"], df_all_bio_s1_spline_u["β"]+std_u, df_all_bio_s1_spline_u["β"]-std_u,\
                     facecolor="gray", alpha=0.8)
    ax3.fill_between(df_all_bio_s1_spline_n["Time"], df_all_bio_s1_spline_n["β"]+std_n, df_all_bio_s1_spline_n["β"]-std_n,\
                     facecolor="gray", alpha=0.8)
    ax_list = [ax1, ax2, ax3]
    for i_num in range(len(ax_list)):
        for i in range(len(REST_START)):
            # REST
            ax_list[i_num].axvspan(REST_START[i], REST_START[i]+24, color=(0, 0, 0.9), alpha=0.2)
            # STIM
            ax_list[i_num].axvspan(STIM_START[i], STIM_START[i]+24, color=(0.9, 0, 0), alpha=0.2)
            # QUES
            ax_list[i_num].axvspan(QUES_START[i], QUES_START[i]+45, color=(0, 0.9, 0), alpha=0.2)
    fig.suptitle("ALL")
    ax1.set_title("pleasant_session1_β")
    ax2.set_title("unpleasant_session1_β")
    ax3.set_title("odorless_session1_β")
    plt.show()

# 変動係数プロット
if alldata_plot == True:
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.plot(df_all_bio_s1_spline_p["Time"], df_all_bio_s1_spline_p["β_cv"], color = "darkgreen")
    ax2.plot(df_all_bio_s1_spline_u["Time"], df_all_bio_s1_spline_u["β_cv"], color = "darkgreen")
    ax3.plot(df_all_bio_s1_spline_n["Time"], df_all_bio_s1_spline_n["β_cv"], color = "darkgreen")
    ax_list = [ax1, ax2, ax3]
    for i_num in range(len(ax_list)):
        for i in range(len(REST_START)):
            # REST
            ax_list[i_num].axvspan(REST_START[i], REST_START[i]+24, color=(0, 0, 0.9), alpha=0.2)
            # STIM
            ax_list[i_num].axvspan(STIM_START[i], STIM_START[i]+24, color=(0.9, 0, 0), alpha=0.2)
            # QUES
            ax_list[i_num].axvspan(QUES_START[i], QUES_START[i]+45, color=(0, 0.9, 0), alpha=0.2)
    fig.suptitle("ALL")
    ax1.set_title("pleasant_session1_β_cv")
    ax2.set_title("unpleasant_session1_β_cv")
    ax3.set_title("odorless_session1_β_cv")
    plt.show()


