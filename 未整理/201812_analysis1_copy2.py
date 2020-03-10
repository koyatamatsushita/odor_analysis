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
import matplotlib.pyplot as plt

# 【前処理】 ####################################################################
# 快臭データ
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
FLAG_NUM_LIST_1 = [[3, 15], [16, 28]]
# 刺激開始フラグ番号
FLAG_NUM_LIST_2 = [4, 7, 10, 13, 17, 20, 23, 26]

# 刺激等の時間(in session)
REST_START = [60, 164, 268, 372]
STIM_START = [84, 188, 292, 396] #STIM_ENDは+24
QUES_START = [113, 217, 321, 425] # QUES_ENDは+45

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
        
        df_diff = df_bio[["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]]
        df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                               columns=["diffμ", "diffη", "diffβ", "diffHR", "diffL/H30", "diffHF30", "diffLF30"])
        
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
                    
        ### セッション毎にデータを切り分ける ###
        df_bio_s1 = df_bio[df_bio["SessionNum"]==1]
        tmp = df_bio_s1["Time"].iloc[0]
        df_bio_s1 = df_bio_s1.assign(Time = df_bio_s1["Time"].sub(tmp))

        df_bio_s2 = df_bio[df_bio["SessionNum"]==2]
        tmp = df_bio_s2["Time"].iloc[0]
        df_bio_s2 = df_bio_s2.assign(Time = df_bio_s2["Time"].sub(tmp))
        
        
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

        

### グラフを描画する ###

"""
# 実験全体
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(list_data_p[0]["Time"], list_data_p[0]["β"], label="pleasant_β")
ax2.plot(list_data_u[0]["Time"], list_data_u[0]["β"], label="unpleasant_β")
ax3.plot(list_data_n[0]["Time"], list_data_n[0]["β"], label="odorless_β")

#ax1.axvspan(100, 1000, color=(0.8, 0, 0), alpha=0.3) # 領域を色付け

fig.suptitle("Sub_A")
ax1.set_title("pleasant_β")
ax2.set_title("unpleasant_β")
ax3.set_title("odorless_β")

plt.tight_layout()
plt.show()
"""


# セッション毎
fig = plt.figure()
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]

ax1.plot(list_data_p_s1[0]["Time"], list_data_p_s1[0]["β"], label="pleasant_session1_β")
ax2.plot(list_data_p_s2[0]["Time"], list_data_p_s2[0]["β"], label="pleasant_session2_β")
ax3.plot(list_data_u_s1[0]["Time"], list_data_u_s1[0]["β"], label="unpleasant_session1_β")
ax4.plot(list_data_u_s2[0]["Time"], list_data_u_s2[0]["β"], label="unpleasant_session2_β")
ax5.plot(list_data_n_s1[0]["Time"], list_data_n_s1[0]["β"], label="odorless_session1_β")
ax6.plot(list_data_n_s2[0]["Time"], list_data_n_s2[0]["β"], label="odorless_session2_β")


# イベント情報込みの場合　以下を記述
ax1_2 = ax1.twinx()
ax2_2 = ax2.twinx()
ax3_2 = ax3.twinx()
ax4_2 = ax4.twinx()
ax5_2 = ax5.twinx()
ax6_2 = ax6.twinx()   
ax1_2.plot(list_data_p_s1[0]["Time"], list_data_p_s1[0]["イベント情報"],\
           label="pleasant_session1_β", color = "orange")
ax2_2.plot(list_data_p_s2[0]["Time"], list_data_p_s2[0]["イベント情報"],\
           label="pleasant_session2_β", color = "orange")
ax3_2.plot(list_data_u_s1[0]["Time"], list_data_u_s1[0]["イベント情報"],\
           label="unpleasant_session1_β", color = "orange")
ax4_2.plot(list_data_u_s2[0]["Time"], list_data_u_s2[0]["イベント情報"],\
           label="unpleasant_session2_β", color = "orange")
ax5_2.plot(list_data_n_s1[0]["Time"], list_data_n_s1[0]["イベント情報"],\
           label="odorless_session1_β", color = "orange")
ax6_2.plot(list_data_n_s2[0]["Time"], list_data_n_s2[0]["イベント情報"],\
           label="odorless_session2_β", color = "orange")


# 領域を色付け
for i_sub in range(len(FILENAME_LIST_BIO[0])):
    for i in range(len(REST_START)):
        # REST
        ax_list[i_sub].axvspan(REST_START[i], REST_START[i]+24, color=(0.8, 0, 0), alpha=0.3)
        # STIM
        ax_list[i_sub].axvspan(STIM_START[i], STIM_START[i]+24, color=(0, 0.8, 0), alpha=0.3)
        # QUES
        ax_list[i_sub].axvspan(QUES_START[i], QUES_START[i]+45, color=(0, 0, 0.8), alpha=0.3)

fig.suptitle("Sub_A")
ax1.set_title("pleasant_session1_β")
ax2.set_title("pleasant_session2_β")
ax3.set_title("unpleasant_session1_β")
ax4.set_title("unpleasant_session2_β")
ax5.set_title("odorless_session1_β")
ax6.set_title("odorless_session2_β")

plt.tight_layout()
plt.show()
