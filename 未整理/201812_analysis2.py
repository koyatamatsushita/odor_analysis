# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:32:35 2019

@author: A_lifePC
stanが付いた指標は標準化してから統計量を計算，
stanがついていない指標は統計量を計算してから標準化
"""

# 【import】 ####################################################################
import os
import copy
import numpy as np
import pandas as pd
import scipy.stats as sp
from scipy import fftpack
from scipy import signal
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import itertools

# 【前処理】 ####################################################################
### 条件に応じて書き換える
data_calculation = False
fc = 0.15 # カットオフ周波数

plot_pca_graph = False

plot_graph = False
# 以下描画する波形
bio_drow = False
stan_drow = False
stim_drow = False
lowfilt_drow = False
highfilt_drow = False
diff_drow = False
lowdiff_drow = False
highdiff_drow = True
# 波形を保存するかどうか
save_plot = False

alldata_plot = False
all_ttest = False

powerspectraｌ = False
stim_rest_ttest = False

plot_sub_list = ["Sub_A", "Sub_B", "Sub_C", "Sub_D", "Sub_E", "Sub_F"]
#plot_bio_list = ["μ", "η", "β", "HR", "L/H30", "HF30", "LF30"]
#plot_bio_list = ["η", "β", "τ", "HR", "L/H30"]
#plot_bio_list = ["η", "β", "τ"]
plot_bio_list = ["η"]

### 以下はデータが変わらん限り固定でOK
DATAPATH = "C:/Users/A_lifePC/Desktop/ファイル読み込み用/2018_11/Unpleasant1"

DATAPATH_SAVE = "C:/Users/A_lifePC/Desktop/ファイル書き出し用/2018_11/plot_graph"

FILENAME_LIST_QUE = ["SubA_question.csv",
                     "SubB_question.csv",
                     "SubC_question.csv",
                     "SubD_question.csv",
                     "SubE_question.csv",
                     "SubF_question.csv"]

FILENAME_LIST_BIO = ["SubA.csv",
                     "SubB.csv",
                     "SubC.csv",
                     "SubD.csv",
                     "SubE.csv",
                     "SubF.csv"]

## 以下のフラグ番号はグラフを見て調べる必要あり
# セッションフラグ番号
FLAG_NUM_LIST_1 = [[0, 4], [4, 8], [9, 13], [13, 17]]
# 刺激開始フラグ番号
FLAG_NUM_LIST_2 = [1, 2, 3, 5, 6, 7, 10, 11, 12, 14, 15, 16]
# 取り除くセッション([[被験者], [タスク番号]])
REMOVE_SESSION = [["Sub_A", "Sub_B", "Sub_C", "Sub_D", "Sub_E", "Sub_F"],
                  [[4], [4], [1, 4], [4], [4], [4]]]

# 刺激等の時間(in session)
REST_START = [60, 163, 267] #REST_ENDは+24
STIM_START = [84, 187, 291] #STIM_ENDは+24
QUES_START = [113, 216, 320] # QUES_ENDは+45

SUB_DICT = {"Sub_A":0, "Sub_B":1,"Sub_C":2,"Sub_D":3,"Sub_E":4,"Sub_F":5}

# 主観評価のカラム名のテンプレ
COLUMNS_QUE = ["kassei", "wakuwaku", "kai", "rirakkusu", "hikassei", "taikutu", "hukai", "iraira"]
# 【関数定義】 ##################################################################
# 与えられたデータを主成分分析して，匂い刺激別に分ける関数 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\    
def mypca(data, odor):
    pca = PCA()
    pca.fit(data)
    
    # 分析結果を基に，データセットを主成分に変換する（主成分スコア）
    transformed = pca.fit_transform(data)
    # 匂い情報を追加
    transformed_odor = np.hstack((transformed, odor))
    # 匂い別に分ける
    un = transformed[np.any(transformed_odor=="Unpleasant", axis=1)]
    non = transformed[np.any(transformed_odor=="Nonodor", axis=1)]
    
    # 因子負荷量を計算
    loading = pca.components_*np.c_[np.sqrt(pca.explained_variance_)]
    
    # データフレーム型に変換
    df_un = pd.DataFrame(un)
    df_non = pd.DataFrame(non)
    
    # 主成分の次元ごとの寄与率を出力する
    print("寄与率:\n", pca.explained_variance_ratio_, "\n")
    
    # 因子負荷量を出力する
    #print("PC1の因子負荷量：\n", loading[0], "\n")
    #print("PC2の因子負荷量：\n", loading[1], "\n")
    
    return transformed, df_un, df_non, loading


# 因子負荷量top3を抜き出す関数
def myloading_top3(loading, columns_name):
    df_loading = pd.Series(loading, index=columns_name)
    print("正の負荷量top3:\n", df_loading.sort_values(ascending=False)[0:3], "\n")
    print("不の負荷量top3:\n", df_loading.sort_values(ascending=True)[0:3], "\n")
    return


# 2次元プロットする関数(データ1，データ2，グラフタイトル，x軸ラベル，y軸ラベル) \\\\\\\\\\\\\\\\\
def my2Dplot(data1, data2, tytle=None, x=None, y=None, save=False):
    
    # グラフ描画サイズを設定する
    plt.figure(figsize=(5, 5))

    # 主成分をプロットする
    plt.scatter(data1[:, 0], data1[:, 1], s=80, c=[0.4, 0.6, 0.9], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.scatter(data2[:, 0], data2[:, 1], s=80, c=[0.5, 0.5, 0.5], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.title(tytle, fontsize=18)
    plt.xlabel(x, fontsize=18)
    plt.ylabel(y, fontsize=18)
    
    # グラフを表示する
    #plt.tight_layout()  # タイトルの被りを防ぐ
    plt.show()
    
    if save == True:
        if os.path.isdir(DATAPATH_SAVE + "/PCA_scatter/β") == False:
            os.makedirs(DATAPATH_SAVE + "/PCA_scatter/β")
        os.chdir(DATAPATH_SAVE + "/PCA_scatter/β")
        if tytle==None:
            print("save_error(No tytle)")
        plt.savefig(tytle + ".png")
    
    plt.close()

# 【メイン処理】 ##################################################################
if data_calculation == True:
    # data読み取り
    list_data_que = []
    list_data_bio = []
    list_data_bio_spline = []
    list_data_s1, list_data_s2, list_data_s3, list_data_s4 = [[] for i_session in range(4)]
    
    os.chdir(DATAPATH)
    for i_sub in range(len(FILENAME_LIST_BIO)):
        
        #### 主観評価 ####
        # encoding="cp932"を付けないと文字コードエラー
        df_que = pd.read_csv(FILENAME_LIST_QUE[i_sub], encoding="cp932")
        df_que.index = ["Task"+str(df_que["No"][i]) for i in range(len(df_que))]
        list_data_que.append(df_que)
                
        
        #### 生体情報 ####
        # encoding="cp932"を付けないと文字コードエラー
        df_bio = pd.read_csv(FILENAME_LIST_BIO[i_sub], encoding="cp932")
        
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
            if allFlag[i_raw] >= 0.2 and allFlag[i_raw] < 0.5:
                smallFlagRaw.append(i_raw)
#        print(smallFlagRaw)
        
        #graph check
#        plt.figure()
#        plt.plot(df_bio["Time"], df_bio["イベント情報"], color = [0, 1, 0])
#        plt.plot(df_bio["Time"][smallFlagRaw], df_bio["イベント情報"][smallFlagRaw],\
#                 color = [1, 0, 0], linestyle = "None", marker= ".")
#        plt.show()
        
        
        ### セッション番号抽出 ###
        session_number = np.zeros(shape=(len(df_bio))) # session1,2がそれぞれ1,2で，それ以外は0
        for i_session in range(len(FLAG_NUM_LIST_1)):
            tmp = smallFlagRaw[FLAG_NUM_LIST_1[i_session][0]]
            while tmp <= smallFlagRaw[FLAG_NUM_LIST_1[i_session][1]]:
                session_number[tmp] = i_session+1
                tmp = tmp+1
                
        df_bio["SessionNum"] = session_number
        
        
        ### セッション毎にデータを切り分ける ###
        df_bio_s1, df_bio_s2, df_bio_s3, df_bio_s4 = [pd.DataFrame([]) for i_session in range(4)]
        list_df_bio = [df_bio_s1, df_bio_s2, df_bio_s3, df_bio_s4]
        for i_session in range(len(list_df_bio)):
            list_df_bio[i_session] = df_bio[df_bio["SessionNum"]==i_session+1]
            tmp = list_df_bio[i_session]["Time"].iloc[0]
            list_df_bio[i_session] = list_df_bio[i_session].assign(Time = list_df_bio[i_session]["Time"].sub(tmp))

        
        ### 補正粘性の算出 ###
        for i_session in range(len(list_df_bio)):
            list_df_bio[i_session]["τ"] = (list_df_bio[i_session]["η"] / list_df_bio[i_session]["β"]).values
        
        
        ### スプライン補間 ###
        bio_list = ["μ", "η", "β", "τ", "HR", "L/H30", "HF30", "LF30"]
        df_bio_spline_s1, df_bio_spline_s2, df_bio_spline_s3, df_bio_spline_s4 = [pd.DataFrame([]) for i_session in range(4)]
        list_df_bio_spline = [df_bio_spline_s1, df_bio_spline_s2, df_bio_spline_s3, df_bio_spline_s4]
        xx = np.arange(0, 369) # 大体1sessionは369[s]
        for i_session in range(len(list_df_bio)):
            list_df_bio_spline[i_session]["Time"] = xx
            for i_bio in bio_list:
                df_tmp = list_df_bio[i_session].dropna(subset=[i_bio])
                list_df_bio_spline[i_session][i_bio] = interp.spline(df_tmp["Time"], df_tmp[i_bio], xx)
        
               
        ### 標準化（セッション毎） ###
        for i_session in range(len(list_df_bio_spline)):
            for bio_name in bio_list:
                tmp = list_df_bio_spline[i_session][bio_name].values
                list_df_bio_spline[i_session][bio_name + "_stan"] = sp.zscore(tmp, ddof=1)
        
        
        ### 微分します###
        bio_list = ["μ", "η", "β", "τ", "HR", "L/H30", "HF30", "LF30",\
                    "μ_stan", "η_stan", "β_stan", "τ_stan", "HR_stan", "L/H30_stan", "HF30_stan", "LF30_stan"]
        for i_session in range(len(list_df_bio_spline)):
            df_diff = list_df_bio_spline[i_session][bio_list]
            df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                                   columns=["μ_diff", "η_diff", "β_diff", "τ_diff", "HR_diff", "L/H30_diff", "HF30_diff", "LF30_diff",\
                                            "μ_stan_diff", "η_stan_diff", "β_stan_diff", "τ_stan_diff", "HR_stan_diff", "L/H30_stan_diff", "HF30_stan_diff", "LF30_stan_diff"])
            list_df_bio_spline[i_session] = pd.concat([list_df_bio_spline[i_session], df_diff], axis=1)


        ### フィルタ処理 ###
        bio_list = ["μ", "η", "β", "τ", "HR", "L/H30", "HF30", "LF30",\
                    "μ_stan", "η_stan", "β_stan", "τ_stan", "HR_stan", "L/H30_stan", "HF30_stan", "LF30_stan"]
        for i_session in range(len(list_df_bio_spline)):
            for bio_name in bio_list:
                ### スムーザ使用 (https://org-technology.com/posts/smoother.html)
                tmp = list_df_bio_spline[i_session][bio_name].values
                n = len(tmp)
                # FFT処理と周波数スケールの作成
                tmpf = fftpack.fft(tmp)/(n/2)
                tmpf[0] = tmpf[0]/2 # 直流成分の振幅を揃える(実用上は不要)
                freq = fftpack.fftfreq(n)
                
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
                
                list_df_bio_spline[i_session][bio_name + "_low"] = tmp_low
                list_df_bio_spline[i_session][bio_name + "_high"] = tmp_high
            
        
        ### ローパスを微分します ###
        bio_list = ["μ_low", "η_low", "β_low", "τ_low", "HR_low", "L/H30_low", "HF30_low", "LF30_low",
                    "μ_stan_low", "η_stan_low", "β_stan_low", "τ_stan_low", "HR_stan_low", "L/H30_stan_low", "HF30_stan_low", "LF30_stan_low"]
        for i_session in range(len(list_df_bio_spline)):
            df_diff = list_df_bio_spline[i_session][bio_list]
            df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                                   columns=["μ_low_diff", "η_low_diff", "β_low_diff", "τ_low_diff", "HR_low_diff", "L/H30_low_diff", "HF30_low_diff", "LF30_low_diff",\
                                            "μ_stan_low_diff", "η_stan_low_diff", "β_stan_low_diff", "τ_stan_low_diff", "HR_stan_low_diff", "L/H30_stan_low_diff", "HF30_stan_low_diff", "LF30_stan_low_diff"])
            list_df_bio_spline[i_session] = pd.concat([list_df_bio_spline[i_session], df_diff], axis=1)
    
        ### ハイパスを微分します ###
        bio_list = ["μ_high", "η_high", "β_high", "τ_high", "HR_high", "L/H30_high", "HF30_high", "LF30_high",
                    "μ_stan_high", "η_stan_high", "β_stan_high", "τ_stan_high", "HR_stan_high", "L/H30_stan_high", "HF30_stan_high", "LF30_stan_high"]
        for i_session in range(len(list_df_bio_spline)):
            df_diff = list_df_bio_spline[i_session][bio_list]
            df_diff = pd.DataFrame(np.insert(np.diff(df_diff.values, axis=0), 0, 0, axis=0),\
                                   columns=["μ_high_diff", "η_high_diff", "β_high_diff", "τ_high_diff", "HR_high_diff", "L/H30_high_diff", "HF30_high_diff", "LF30_high_diff",\
                                            "μ_stan_high_diff", "η_stan_high_diff", "β_stan_high_diff", "τ_stan_high_diff", "HR_stan_high_diff", "L/H30_stan_high_diff", "HF30_stan_high_diff", "LF30_stan_high_diff"])
            list_df_bio_spline[i_session] = pd.concat([list_df_bio_spline[i_session], df_diff], axis=1)
        
        # リストにデータ保存
        list_data_bio.append(list_df_bio)
        list_data_bio_spline.append(list_df_bio_spline)
        list_data_s1.append(list_df_bio_spline[0])
        list_data_s2.append(list_df_bio_spline[1])
        list_data_s3.append(list_df_bio_spline[2])
        list_data_s4.append(list_df_bio_spline[3])


    # 刺激中のみを抽出
    list_data_bio_spline_stim = []
    for i_sub in range(len(list_data_bio_spline)):
        tmp = []
        for i_session in range(len(FLAG_NUM_LIST_1)):
            for i_task in range(len(STIM_START)):
                tmp.append(list_data_bio_spline[i_sub][i_session][(list_data_bio_spline[i_sub][i_session]["Time"]>=STIM_START[i_task]) & (list_data_bio_spline[i_sub][i_session]["Time"]<STIM_START[i_task]+24)])
        list_data_bio_spline_stim.append(tmp)
    
    
    # 刺激中の最大，最小，平均，標準偏差
    bio_list = ["μ", "η", "β", "τ", "HR", "L/H30", "HF30", "LF30",\
                "μ_diff", "η_diff", "β_diff", "τ_diff", "HR_diff", "L/H30_diff", "HF30_diff", "LF30_diff",\
                "μ_stan", "η_stan", "β_stan", "τ_stan", "HR_stan", "L/H30_stan", "HF30_stan", "LF30_stan",\
                "μ_stan_diff", "η_stan_diff", "β_stan_diff", "τ_stan_diff", "HR_stan_diff", "L/H30_stan_diff", "HF30_stan_diff", "LF30_stan_diff",\
                "μ_low", "η_low", "β_low", "τ_low", "HR_low", "L/H30_low", "HF30_low", "LF30_low",\
                "μ_stan_low", "η_stan_low", "β_stan_low", "τ_stan_low", "HR_stan_low", "L/H30_stan_low", "HF30_stan_low", "LF30_stan_low",\
                "μ_high", "η_high", "β_high", "τ_high", "HR_high", "L/H30_high", "HF30_high", "LF30_high",\
                "μ_stan_high", "η_stan_high", "β_stan_high", "τ_stan_high", "HR_stan_high", "L/H30_stan_high", "HF30_stan_high", "LF30_stan_high",\
                "μ_low_diff", "η_low_diff", "β_low_diff", "τ_low_diff", "HR_low_diff", "L/H30_low_diff", "HF30_low_diff", "LF30_low_diff",\
                "μ_stan_low_diff", "η_stan_low_diff", "β_stan_low_diff", "τ_stan_low_diff", "HR_stan_low_diff", "L/H30_stan_low_diff", "HF30_stan_low_diff", "LF30_stan_low_diff",\
                "μ_high_diff", "η_high_diff", "β_high_diff", "τ_high_diff", "HR_high_diff", "L/H30_high_diff", "HF30_high_diff", "LF30_high_diff",\
                "μ_stan_high_diff", "η_stan_high_diff", "β_stan_high_diff", "τ_stan_high_diff", "HR_stan_high_diff", "L/H30_stan_high_diff", "HF30_stan_high_diff", "LF30_stan_high_diff"]
    list_data_bio_spline_stim_stat = []
    for i_sub in range(len(list_data_bio_spline)):
        list_tmp = []
        for bio_name in bio_list:
            df = pd.DataFrame()
            for i_task in range(len(list_data_bio_spline_stim[i_sub])):
                tmp_max = np.max(list_data_bio_spline_stim[i_sub][i_task][bio_name].values)
                tmp_min = np.min(list_data_bio_spline_stim[i_sub][i_task][bio_name].values)
                tmp_mean = np.mean(list_data_bio_spline_stim[i_sub][i_task][bio_name].values)
                tmp_std = np.std(list_data_bio_spline_stim[i_sub][i_task][bio_name].values, ddof=1)
                df["Task"+str(i_task+1)] = tmp_max, tmp_min, tmp_mean, tmp_std
            df.index = [bio_name+"_max", bio_name+"_min", bio_name+"_mean", bio_name+"_std"]
            list_tmp.append(df.T)
        list_data_bio_spline_stim_stat.append(list_tmp)
    
    
    # セッションの削除
    list_data_que_original = copy.deepcopy(list_data_que)
    for i_sub in range(len(REMOVE_SESSION[0])):
        del_sub = REMOVE_SESSION[0][i_sub]
        list_tmp = REMOVE_SESSION[1][i_sub]
        list_del = []
        for i_del in range(len(list_tmp)):
            i_tmp = list_tmp[i_del]-1
            list_del.extend([i_tmp*3+i for i in range(1, 4)])
        # 生体情報削除
        for i_bio in range(len(bio_list)):
            list_data_bio_spline_stim_stat[SUB_DICT[del_sub]][i_bio].drop(index=["Task"+str(i) for i in list_del], inplace=True)
        # 主観評価削除
        list_data_que[SUB_DICT[del_sub]] = list_data_que[SUB_DICT[del_sub]].drop(index=["Task"+str(i) for i in list_del])
        
    
    # 被験者内で標準化
    list_data_que_stan = []
    for i_sub in range(len(list_data_que)):
        df_que_stan = list_data_que[i_sub].drop(["No", "Stimulation", "Intensity"], axis=1)
        df_que_stan = pd.DataFrame(sp.zscore(df_que_stan.values, axis=0, ddof=1), index=df_que_stan.index)
        list_data_que_stan.append(df_que_stan)
    
    #!!! 生体情報はタスク内で標準化したデータがあるので，この作業は不要 !!!    
    list_data_bio_spline_stim_stat_stan = []
    for i_sub in range(len(list_data_bio_spline_stim_stat)):
        df = pd.DataFrame()
        list_tmp = []
        for i_bio in range(len(bio_list)):
            for i_task in range(len(list_data_bio_spline_stim_stat[i_sub][i_bio])):
                df = list_data_bio_spline_stim_stat[i_sub][i_bio].apply(sp.zscore, axis=0, ddof=1)
            list_tmp.append(df)
        list_data_bio_spline_stim_stat_stan.append(list_tmp)
    #list_data_bio_spline_stim_stat_stan = copy.deepcopy(list_data_bio_spline_stim_stat)
    
    
    # 不快臭と無臭を切り分ける
    df_data_que_unpleasant, df_data_que_odorless = [pd.DataFrame() for i in range(2)]
    list_data_bio_spline_stim_stat_stan_unpleasant, list_data_bio_spline_stim_stat_stan_odorless = [[] for i in range(2)] 
    df_data_bio_spline_stim_stat_stan_unpleasant, df_data_bio_spline_stim_stat_stan_odorless = [pd.DataFrame() for i in range(2)]
    for odor in ["Unpleasant", "Nonodor"]:
        for i_sub in range(len(list_data_bio_spline_stim_stat_stan)):
            unpleasant = list(list_data_que[i_sub][list_data_que[i_sub]["Stimulation"] == odor]["No"].values)
            df_sub_que, df_sub_bio = [pd.DataFrame() for i in range(2)]
            for i_bio in range(len(bio_list)):
                list_df = []
                for i_un in range(len(unpleasant)):
                    list_df.append(list_data_bio_spline_stim_stat_stan[i_sub][i_bio].loc["Task"+str(unpleasant[i_un])])
                df_bio = pd.DataFrame()
                for i_un in range(len(unpleasant)):
                    df_bio = pd.concat([df_bio, list_df[i_un]], axis=1, sort=False)
                df_sub_bio = pd.concat([df_sub_bio, df_bio.T], axis=1, sort=False)
            df_sub_que = list_data_que_stan[i_sub].loc[["Task"+str(unpleasant[i]) for i in range(len(unpleasant))]]
            
            if odor == "Unpleasant":
                df_data_que_unpleasant = pd.concat([df_data_que_unpleasant, df_sub_que], axis=0, sort=False)
                list_data_bio_spline_stim_stat_stan_unpleasant.append(df_sub_bio)
                df_data_bio_spline_stim_stat_stan_unpleasant = pd.concat([df_data_bio_spline_stim_stat_stan_unpleasant, df_sub_bio], axis=0, sort=False)
            elif odor == "Nonodor":
                df_data_que_odorless = pd.concat([df_data_que_odorless, df_sub_que], axis=0, sort=False)
                list_data_bio_spline_stim_stat_stan_odorless.append(df_sub_bio)
                df_data_bio_spline_stim_stat_stan_odorless = pd.concat([df_data_bio_spline_stim_stat_stan_odorless, df_sub_bio], axis=0, sort=False)



# 主成分分析
# 主観評価
df_data_que_unpleasant["Stim"] = "Unpleasant"
df_data_que_odorless["Stim"] = "Nonodor"
df_data_que_for_pca = pd.concat([df_data_que_unpleasant, df_data_que_odorless], axis=0, sort=False)
data_que_for_pca = df_data_que_for_pca.drop("Stim", axis=1).values
data_que_for_pca = sp.zscore(data_que_for_pca, axis=0, ddof=1) # 標準化の標準化
odor = df_data_que_for_pca["Stim"].values.tolist()
odor = np.reshape(odor, [len(odor), 1])
print("主観評価のPCA----------------------------------------------------------")
transformed_que, df_un_que, df_non_que, loading_que = mypca(data_que_for_pca, odor)
print("主観評価PC1")
myloading_top3(loading_que[0], COLUMNS_QUE) # pc1の因子負荷量
print("主観評価PC2")
myloading_top3(loading_que[1], COLUMNS_QUE) # pc2の因子負荷量


# 生体情報
df_data_bio_spline_stim_stat_stan_unpleasant["Stim"] = "Unpleasant"
df_data_bio_spline_stim_stat_stan_odorless["Stim"] = "Nonodor"
df_data_bio_for_pca = pd.concat([df_data_bio_spline_stim_stat_stan_unpleasant, df_data_bio_spline_stim_stat_stan_odorless], axis=0, sort=False)

#bio_name_for_pca = ["η_low", "β_low","HR_low", "L/H30_low",  "η_low_diff", "β_low_diff","HR_low_diff", "L/H30_low_diff", \
#                    "η_high", "β_high","HR_high", "L/H30_high",  "η_high_diff", "β_high_diff", "HR_high_diff", "L/H30_high_diff", ]
bio_name_for_pca = ["β_low", "β_low_diff",\
                    "β_high", "β_high_diff"]

list_bio_name_for_pca = []
for i_bio in range(len(bio_name_for_pca)):
    tmp = bio_name_for_pca[i_bio]
    list_bio_name_for_pca.extend([tmp+"_max", tmp+"_min", tmp+"_mean", tmp+"_std"])

data_bio_for_pca = df_data_bio_for_pca.loc[:, list_bio_name_for_pca].values
data_bio_for_pca = sp.zscore(data_bio_for_pca, axis=0, ddof=1) # 標準化の標準化
#data_bio_for_pca = df_data_bio_for_pca.drop("Stim", axis=1).values
#bio_list = df_data_bio_for_pca.drop("Stim", axis=1).columns
odor = df_data_bio_for_pca["Stim"].values.tolist()
odor = np.reshape(odor, [len(odor), 1])
print("生体情報のPCA----------------------------------------------------------")
transformed_bio, df_un_bio, df_non_bio, loading_bio = mypca(data_bio_for_pca, odor)
print("生体情報PC1")
myloading_top3(loading_bio[0], list_bio_name_for_pca) # pc1の因子負荷量
#print("生体情報PC2")
#myloading_top3(loading_bio[1], list_bio_name_for_pca) # pc2の因子負荷量


# データ変数の準備
# 主観評価PC1
score1_que = np.hstack((df_un_que.iloc[:, 0].values, df_non_que.iloc[:, 0].values))
un1_que = df_un_que.iloc[:, 0].values
non1_que = df_non_que.iloc[:, 0].values
# 主観評価PC2
score2_que = np.hstack((df_un_que.iloc[:, 1].values, df_non_que.iloc[:, 1].values))
un2_que = df_un_que.iloc[:, 1].values
non2_que = df_non_que.iloc[:, 1].values
# 生体情報PC1
score1_bio = np.hstack((df_un_bio.iloc[:, 0].values, df_non_bio.iloc[:, 0].values))
un1_bio = df_un_bio.iloc[:, 0].values
non1_bio = df_non_bio.iloc[:, 0].values
# 生体情報PC2
score2_bio = np.hstack((df_un_bio.iloc[:, 1].values, df_non_bio.iloc[:, 1].values))
un2_bio = df_un_bio.iloc[:, 1].values
non2_bio = df_non_bio.iloc[:, 1].values


# 匂い間の有意差検定(Welch T Test)
#print("主観評価PC1_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(un1_que, non1_que, equal_var=False), "\n")
#print("主観評価PC2_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(un2_que, non2_que, equal_var=False), "\n")
#print("生体情報PC1_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(un1_bio, non1_bio, equal_var=False), "\n")
#print("生体情報PC2_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(un2_bio, non2_bio, equal_var=False), "\n")

print("主観評価PC1_不快と無臭のt検定p値：\n", sp.stats.ttest_ind(un1_que, non1_que, equal_var=False)[1], "\n")
print("生体情報PC1_不快と無臭のt検定p値：\n", sp.stats.ttest_ind(un1_bio, non1_bio, equal_var=False)[1], "\n")


# 相関を計算
print("主観評価PC1と生体情報PC1の相関を解析----------------------------------")  
# 相関係数，p値を計算する
cor = sp.stats.pearsonr(score1_que, score1_bio)
print("相関係数：\n", cor[0])
print("相関のp値：\n", cor[1], "\n")

# 作図用
bio_name_for_tytle = "_".join(bio_name_for_pca)
bio_name_for_tytle = "_".join(["cor="+str(round(cor[0], 3)), bio_name_for_tytle])



# 生体情報を総当たりして，フォルダに画像を保存する !!! 実行時間超長いよ !!!
#bio_name_for_pca = ["μ", "η", "β", "τ", "HR", "L/H30", "HF30", "LF30",\
#                    "μ_low", "η_low", "β_low", "τ_low", "HR_low", "L/H30_low", "HF30_low", "LF30_low",\
#                    "μ_high", "η_high", "β_high", "τ_high", "HR_high", "L/H30_high", "HF30_high", "LF30_high",\
#                    "μ_diff", "η_diff", "β_diff", "τ_diff", "HR_diff", "L/H30_diff", "HF30_diff", "LF30_diff",\
#                    "μ_low_diff", "η_low_diff", "β_low_diff", "τ_low_diff", "HR_low_diff", "L/H30_low_diff", "HF30_low_diff", "LF30_low_diff",\
#                    "μ_high_diff", "η_high_diff", "β_high_diff", "τ_high_diff", "HR_high_diff", "L/H30_high_diff", "HF30_high_diff", "LF30_high_diff"]
#bio_name_for_pca = ["μ_high", "η_high", "β_high", "τ_high", "HR_high", "L/H30_high",\
#                    "μ_high_diff", "η_high_diff", "β_high_diff", "τ_high_diff", "HR_high_diff", "L/H30_high_diff"]
#bio_name_for_pca = ["μ_low", "η_low", "β_low", "τ_low", "HR_low", "L/H30_low",\
#                    "μ_low_diff", "η_low_diff", "β_low_diff", "τ_low_diff", "HR_low_diff", "L/H30_low_diff"]
bio_name_for_pca = ["β_low",\
                    "β_high",\
                    "β_low_diff",\
                    "β_high_diff"]

"""
bio_name_for_pca2 = []
for i_bio in range(len(bio_name_for_pca)):    
    tmp = bio_name_for_pca[i_bio]
    bio_name_for_pca2.extend([tmp+"_max", tmp+"_min", tmp+"_mean", tmp+"_std"])
# bio_name_for_pca2内の要素の4指標の相関が0.4以下の指標を削除
bio_name_for_pca = []
for i_bio2 in range(len(bio_name_for_pca2)):
    # PCA
    data_bio_for_pca_tmp = df_data_bio_for_pca.loc[:, bio_name_for_pca2[i_bio2]].values
    cor = sp.stats.pearsonr(score1_que, data_bio_for_pca_tmp)[0] # 相関係数
    if cor > 0.5:
        bio_name_for_pca.append(bio_name_for_pca2[i_bio2])

# bio_name_for_pca内の要素の総当たりで主成分分析
for i_bio in range(1, len(bio_name_for_pca)+1):
    list_combinations = list(itertools.combinations(bio_name_for_pca, i_bio))
    for i_bio2 in range(len(list_combinations)):
        list_bio_combinations = []
        for i_bio3 in range(len(list_combinations[i_bio2])):
            tmp = list_combinations[i_bio2][i_bio3]
            list_bio_combinations.append(tmp)
        # 総当たりPCA
        data_bio_for_pca_tmp = df_data_bio_for_pca.loc[:, list_bio_combinations].values
        transformed_bio_tmp, df_un_bio_tmp, df_non_bio_tmp, loading_bio_tmp = mypca(data_bio_for_pca_tmp, odor)
        score1_bio_tmp = np.hstack((df_un_bio_tmp.iloc[:, 0].values, df_non_bio_tmp.iloc[:, 0].values))
        un1_bio_tmp = df_un_bio_tmp.iloc[:, 0].values
        non1_bio_tmp = df_non_bio_tmp.iloc[:, 0].values
        pvalue_ttest = sp.stats.ttest_ind(un1_bio_tmp, non1_bio_tmp, equal_var=False)[1] # pvalue
        cor = sp.stats.pearsonr(score1_que, score1_bio_tmp)[0] # 相関係数
        # PCA結果を描画する
        bio_name_for_tytle = "_".join(list_combinations[i_bio2])
        bio_name_for_tytle = "_".join(["cor="+str(round(cor, 3)), bio_name_for_tytle])
        bio_name_for_tytle = bio_name_for_tytle.replace("/", "") # スラッシュがある場合には削除
        my2Dplot(np.vstack((un1_que, un1_bio_tmp)).T, np.vstack((non1_que, non1_bio_tmp)).T,\
                 bio_name_for_tytle, "Subjective evaluation pc1", "Biometric information pc1", save=True)
"""


# bio_name_for_pca内の要素の総当たりで主成分分析
for i_bio in range(1, len(bio_name_for_pca)+1):
    list_combinations = list(itertools.combinations(bio_name_for_pca, i_bio))
    for i_bio2 in range(len(list_combinations)):
        list_bio_combinations = []
        for i_bio3 in range(len(list_combinations[i_bio2])):
            tmp = "".join(list_combinations[i_bio2][i_bio3])
            list_bio_combinations.extend([tmp+"_max", tmp+"_min", tmp+"_mean", tmp+"_std"])
        # 総当たりPCA
        data_bio_for_pca_tmp = df_data_bio_for_pca.loc[:, list_bio_combinations].values
        transformed_bio_tmp, df_un_bio_tmp, df_non_bio_tmp, loading_bio_tmp = mypca(data_bio_for_pca_tmp, odor)
        score1_bio_tmp = np.hstack((df_un_bio_tmp.iloc[:, 0].values, df_non_bio_tmp.iloc[:, 0].values))
        un1_bio_tmp = df_un_bio_tmp.iloc[:, 0].values
        non1_bio_tmp = df_non_bio_tmp.iloc[:, 0].values
        pvalue_ttest = sp.stats.ttest_ind(un1_bio_tmp, non1_bio_tmp, equal_var=False)[1] # pvalue
        cor = sp.stats.pearsonr(score1_que, score1_bio_tmp)[0] # 相関係数
        # PCA結果を描画する
        bio_name_for_tytle = "_".join(list_combinations[i_bio2])
        bio_name_for_tytle = "_".join(["cor="+str(round(cor, 3)), bio_name_for_tytle])
        bio_name_for_tytle = bio_name_for_tytle.replace("/", "") # スラッシュがある場合には削除
        my2Dplot(np.vstack((un1_que, un1_bio_tmp)).T, np.vstack((non1_que, non1_bio_tmp)).T,\
                 bio_name_for_tytle, "Subjective evaluation pc1", "Biometric information pc1", save=True)


"""
# 生体情報を総当たりして，有意差p値が小さく，相関係数が大きい組み合わせを探す
list_all_combi = []
list_best_combi = []
best_cor = 0

# bio index
#list_bio_combinations = ["β_max", "β_low_max", "β_high_max", "β_diff_max", "β_low_diff_max", "β_high_diff_max",
#                         "β_min", "β_low_min", "β_high_min", "β_diff_min", "β_low_diff_min", "β_high_diff_min",
#                         "β_mean", "β_low_mean", "β_high_mean", "β_diff_mean", "β_low_diff_mean", "β_high_diff_mean",
#                         "β_std", "β_low_std", "β_high_std", "β_diff_std", "β_low_diff_std", "β_high_diff_std"]
#list_bio_combinations = ["β_max", "β_min", "β_mean", "β_std"]
list_bio_combinations = ["η_max", "η_diff_max", "η_min", "η_diff_min", "η_mean", "η_diff_mean", "η_std", "η_diff_std",
                         "β_max", "β_diff_max", "β_min", "β_diff_min", "β_mean", "β_diff_mean", "β_std", "β_diff_std",
                         "τ_max", "τ_diff_max", "τ_min", "τ_diff_min", "τ_mean", "τ_diff_mean", "τ_std", "τ_diff_std"]

for i_bio in range(1, len(list_bio_combinations)+1):
    list_combinations = list(itertools.combinations(list_bio_combinations, i_bio))
    for i_com in range(len(list_combinations)):
        data_bio_for_pca_tmp = df_data_bio_for_pca.loc[:, list(list_combinations[i_com])].values
        transformed_bio_tmp, df_un_bio_tmp, df_non_bio_tmp, loading_bio_tmp = mypca(data_bio_for_pca_tmp, odor)
        score1_bio_tmp = np.hstack((df_un_bio_tmp.iloc[:, 0].values, df_non_bio_tmp.iloc[:, 0].values))
        un1_bio_tmp = df_un_bio_tmp.iloc[:, 0].values
        non1_bio_tmp = df_non_bio_tmp.iloc[:, 0].values
        pvalue_ttest = sp.stats.ttest_ind(un1_bio_tmp, non1_bio_tmp, equal_var=False)[1] # pvalue
        cor = sp.stats.pearsonr(score1_que, score1_bio_tmp)[0] # 相関係数
        if (pvalue_ttest<0.05 and cor>0.55):
            list_best_combi.append(list_combinations[i_com])
            print(str(list_combinations[i_com])+"\npvalue: "+str(pvalue_ttest)+"\ncor: "+str(cor)+"\n")
            if cor > best_cor:
                best_cor = cor
                best_combi = list_combinations[i_com]
        list_all_combi.append(list_combinations[i_com])
        print(str(list(list_combinations[i_com]))+"\npvalue: "+str(pvalue_ttest)+"\ncor: "+str(cor)+"\n")
        
        #my2Dplot(np.vstack((un1_que, un1_bio_tmp)).T, np.vstack((non1_que, non1_bio_tmp)).T,\
        #         "PCA", "Subjective evaluation pc1", "Biometric information pc1")
print(str(best_cor)+"\n"+str(best_combi))
"""



### グラフを描画する ###
if plot_graph == True:
    
    # セッション毎
    for bio_name in plot_bio_list:
        for i_sub in plot_sub_list:
            unpleasant = list(list_data_que_original[SUB_DICT[i_sub]][list_data_que_original[SUB_DICT[i_sub]]["Stimulation"] == "Unpleasant"]["No"].values)
                                    
            tytle = ""
            fig = plt.figure(figsize=(18,9))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            
            list_ax = [ax1, ax2, ax3, ax4]
            
            if bio_drow == True:
                for i_session in range(len(list_ax)):
                    list_ax[i_session].plot(list_data_bio_spline[SUB_DICT[i_sub]][i_session]["Time"], list_data_bio_spline[SUB_DICT[i_sub]][i_session][bio_name],\
                                            label="pleasant_session1_" + bio_name, color = "darkgreen")                
            
            if stan_drow == True:
                for i_session in range(len(list_ax)):
                    list_ax[i_session].plot(list_data_bio_spline[SUB_DICT[i_sub]][i_session]["Time"], list_data_bio_spline[SUB_DICT[i_sub]][i_session][bio_name + "_stan"],\
                                            label="pleasant_session1_" + bio_name, color = "darkgreen")
                tytle = tytle + "_stan"
            
            # イベント情報込みの場合　以下を記述
            if stim_drow == True:
                list_ax_2 = []
                
                for i_session in range(len(list_ax)):
                    list_ax_2.append(list_ax[i_session].twinx())
                
                for i_session in range(len(list_ax)):
                    list_ax_2[i_session].plot(list_data_bio[SUB_DICT[i_sub]][i_session]["Time"], list_data_bio[SUB_DICT[i_sub]][i_session]["イベント情報"],\
                                              label="pleasant_session1_" + bio_name, color = "orange")
                tytle = tytle + "_event"
    
            
            if lowfilt_drow == True:
                for i_session in range(len(list_ax)):
                    list_ax[i_session].plot(list_data_bio_spline[SUB_DICT[i_sub]][i_session]["Time"], list_data_bio_spline[SUB_DICT[i_sub]][i_session][bio_name + "_low"],\
                                            label="pleasant_session1_" + bio_name + "_low " + str(fc) + "[Hz]", color = "royalblue")
                tytle = tytle + "_low"
                
            if highfilt_drow == True:
                for i_session in range(len(list_ax)):
                    list_ax[i_session].plot(list_data_bio_spline[SUB_DICT[i_sub]][i_session]["Time"], list_data_bio_spline[SUB_DICT[i_sub]][i_session][bio_name + "_high"],\
                                            label="pleasant_session1_" + bio_name + "_high " + str(fc) + "[Hz]", color = "darkred")
                tytle = tytle + "_high"
            
            
            if diff_drow == True:
                for i_session in range(len(list_ax)):
                    list_ax[i_session].plot(list_data_bio_spline[SUB_DICT[i_sub]][i_session]["Time"], list_data_bio_spline[SUB_DICT[i_sub]][i_session][bio_name + "_diff"],\
                                            label="pleasant_session1_" + bio_name + "_diff", color = "black")
                tytle = tytle + "_diff"

    
            if lowdiff_drow == True:
                for i_session in range(len(list_ax)):
                    list_ax[i_session].plot(list_data_bio_spline[SUB_DICT[i_sub]][i_session]["Time"], list_data_bio_spline[SUB_DICT[i_sub]][i_session][bio_name + "_low_diff"],\
                                            label="pleasant_session1_" + bio_name + "_low_diff", color = "gray")
                tytle = tytle + "_lowdiff"
                

            if highdiff_drow == True:
                for i_session in range(len(list_ax)):
                    list_ax[i_session].plot(list_data_bio_spline[SUB_DICT[i_sub]][i_session]["Time"], list_data_bio_spline[SUB_DICT[i_sub]][i_session][bio_name + "_high_diff"],\
                                            label="pleasant_session1_" + bio_name + "_high_diff", color = "black")
                tytle = tytle + "_highdiff"
            
            ymax = max([ax1.axis()[3], ax2.axis()[3], ax3.axis()[3], ax4.axis()[3]])
            ymin = min([ax1.axis()[2], ax2.axis()[2], ax3.axis()[2], ax4.axis()[2]])
            for ax in list_ax:
                ax.set_xlim(0, 370)
                ax.set_ylim(ymin, ymax) 
            
            # 領域を色付け
            for i_num in range(len(list_ax)):
                for i in range(len(REST_START)):
                    # REST
                    #list_ax[i_num].axvspan(REST_START[i], REST_START[i]+24, color=(0.5, 0.5, 0.5), alpha=0.2)
                    # STIM
                    if (3*i_num+i+1 in unpleasant) == True:
                        list_ax[i_num].axvspan(STIM_START[i], STIM_START[i]+24, color=(0, 0, 0.9), alpha=0.5)
                    else:
                        list_ax[i_num].axvspan(STIM_START[i], STIM_START[i]+24, color=(0.1, 0.1, 0.1), alpha=0.5)
                    # QUES
                    #list_ax[i_num].axvspan(QUES_START[i], QUES_START[i]+45, color=(0, 0.9, 0), alpha=0.2)
            
            fig.suptitle(i_sub, fontsize=20)
            for i_session in range(len(list_ax)):
                list_ax[i_session].set_title("session" + str(i_session+1) + "_" + bio_name + tytle, fontsize=10)
                        
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



# PCA結果を描画する
if plot_pca_graph == True:
    my2Dplot(np.vstack((un1_que, un1_bio)).T, np.vstack((non1_que, non1_bio)).T,\
             bio_name_for_tytle, "Subjective evaluation pc1", "Biometric information pc1", save=False)
    # 主観評価PC2まで
    #my2Dplot(np.vstack((un1_que, un2_que)).T, np.vstack((non1_que, non2_que)).T,
    #         "PCA scatter", "Subjective evaluation pc1", "Subjective evaluation pc2")
    # 生体情報PC2まで
    my2Dplot(np.vstack((un1_bio, un2_bio)).T, np.vstack((non1_bio, non2_bio)).T,
             bio_name_for_tytle, "Biometric information pc1", "Biometric information pc2")



# 自由帳#######################################################################
"""
# 任意のグラフを描画しまーす
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax.plot(list_data_bio_spline[2][2]["Time"], list_data_bio_spline[2][2]["η_high_diff"], color = "black")

# 領域を色付け
ax.axvspan(STIM_START[0], STIM_START[0]+24, color=(0, 0, 0.9), alpha=0.5) # 不快臭
ax.axvspan(STIM_START[1], STIM_START[1]+24, color=(0.1, 0.1, 0.1), alpha=0.5) # 無臭
ax.axvspan(STIM_START[2], STIM_START[2]+24, color=(0.1, 0.1, 0.1), alpha=0.5) # 無臭

fig.suptitle("Sub_C", fontsize=20)
ax.set_title("session3_η_high_diff", fontsize=15)
            
plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()
"""



"""
list_bio_name_for_pca = ["η_max", "η_diff_max", "η_min", "η_diff_min", "η_mean", "η_diff_mean", "η_std", "η_diff_std",
                         "β_max", "β_diff_max", "β_min", "β_diff_min", "β_mean", "β_diff_mean", "β_std", "β_diff_std",
                         "τ_max", "τ_diff_max", "τ_min", "τ_diff_min", "τ_mean", "τ_diff_mean", "τ_std", "τ_diff_std",
                         "HR_max", "HR_diff_max", "HR_min", "HR_diff_min", "HR_mean", "HR_diff_mean", "HR_std", "HR_diff_std",
                         "L/H30_max", "L/H30_diff_max", "L/H30_min", "L/H30_diff_min", "L/H30_mean", "L/H30_diff_mean", "L/H30_std", "L/H30_diff_std",
                         "LF30_max", "LF30_diff_max", "LF30_min", "LF30_diff_min", "LF30_mean", "LF30_diff_mean", "LF30_std", "LF30_diff_std",
                         "HF30_max", "HF30_diff_max", "HF30_min", "HF30_diff_min", "HF30_mean", "HF30_diff_mean", "HF30_std", "HF30_diff_std"]
list_bio_name_for_pca = ["η_stan_max", "η_stan_diff_max", "η_stan_min", "η_stan_diff_min", "η_stan_mean", "η_stan_diff_mean", "η_stan_std", "η_stan_diff_std",
                         "β_stan_max", "β_stan_diff_max", "β_stan_min", "β_stan_diff_min", "β_stan_mean", "β_stan_diff_mean", "β_stan_std", "β_stan_diff_std",
                         "HR_stan_max", "HR_stan_diff_max", "HR_stan_min", "HR_stan_diff_min", "HR_stan_mean", "HR_stan_diff_mean", "HR_stan_std", "HR_stan_diff_std",
                         "L/H30_stan_max", "L/H30_stan_diff_max", "L/H30_stan_min", "L/H30_stan_diff_min", "L/H30_stan_mean", "L/H30_stan_diff_mean", "L/H30_stan_std", "L/H30_stan_diff_std",
                         "LF30_stan_max", "LF30_stan_diff_max", "LF30_stan_min", "LF30_stan_diff_min", "LF30_stan_mean", "LF30_stan_diff_mean", "LF30_stan_std", "LF30_stan_diff_std",
                         "HF30_stan_max", "HF30_stan_diff_max", "HF30_stan_min", "HF30_stan_diff_min", "HF30_stan_mean", "HF30_stan_diff_mean", "HF30_stan_std", "HF30_stan_diff_std"]
"""
