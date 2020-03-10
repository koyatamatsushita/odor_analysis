# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

生体情報のデータを読み込み，
タスクごとに生体情報のmean, max, min, stdを計算，
不要なタスクのデータを削除して
各指標毎に標準化，excel出力するプログラム
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats

# 【前処理】 ####################################################################
# セッション数
SESSION_LIST = [4, 4, 4, 4, 4, 4]
# セッションごとのタスク数
TASK_LIST = [3, 3, 3, 3, 3, 3]

# 削除する不要なタスクのタスク番号リスト
DEL_LIST = [[10, 11, 12],
       [10, 11, 12],
       [1, 2, 3, 10, 11, 12],
       [10, 11, 12],
       [10, 11, 12],
       [10, 11, 12]]

# データがあるパスとそのファイル名
DATAPATH = "C:\\Users\\A_lifePC\\Desktop\\ファイル読み込み用\\2018_11\\Unpleasant1"
FILENAME_LIST = ["SubA.xlsx",
                 "SubB.xlsx",
                 "SubC.xlsx",
                 "SubD.xlsx",
                 "SubE.xlsx",
                 "SubF.xlsx"]

# Excelファイルとして書き出すパスとファイル名
DATAPATH2 = "C:\\Users\\A_lifePC\\Desktop\\ファイル書き出し用\\2018_11"
FILENAME_LIST2 = ["SubA_standardize.xlsx",
                  "SubB_standardize.xlsx",
                  "SubC_standardize.xlsx",
                  "SubD_standardize.xlsx",
                  "SubE_standardize.xlsx",
                  "SubF_standardize.xlsx",]

 # 列ラベルテンプレ
COLUMN_TEMP1 = ['Event', 'Time', 'Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Etadiff', 'Betadiff', 'HRdiff', 'LF/HFdiff', 'HFdiff', 'LFdiff']
COLUMN_TEMP2 = ['Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Etadiff', 'Betadiff', 'HRdiff', 'LF/HFdiff', 'HFdiff', 'LFdiff']
COLUMN_TEMP3 = ['Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Etadiff', 'Betadiff', 'HRdiff', 'LF/HFdiff', 'HFdiff', 'LFdiff', 'Task']

# 【メイン処理】 ##################################################################
def main():
    for i_sub in range(len(FILENAME_LIST)): # 被験者数だけ繰り返す
        
        #　データがあるパスに作業ディレクトリ変更
        os.chdir(DATAPATH)
        # データフレームにデータ格納
        data = pd.read_excel(FILENAME_LIST[i_sub])
        
        # 時間データを取り出す
        t = data["Time"]
        # 微分する対象のデータ列を取り出す
        df = pd.DataFrame({"η": data["η"],
                          "β": data["β"],
                          "HR": data["HR"],
                          "LF/HF": data["L/H30"],
                          "HF": data["HF30"],
                          "LF": data["LF30"],})
         
        # differential関数で微分
        diffdf = diff_df(df, t)
        
        # 生体情報のデータフレーム
        data_df = pd.DataFrame({"Time": t,
                               "Event": data["イベント情報"],
                               "Eta": df["η"],
                               "Etadiff": diffdf["η"],
                               "Beta": df["β"],
                               "Betadiff": diffdf["β"],
                               "HR": df["HR"],
                               "HRdiff": diffdf["HR"],
                               "LF/HF": df["LF/HF"],
                               "LF/HFdiff": diffdf["LF/HF"],
                               "HF": df["HF"],
                               "HFdiff": diffdf["HF"],
                               "LF": df["LF"],
                               "LFdiff": diffdf["LF"]})
        
        data_df = data_df.loc[:, COLUMN_TEMP1] # 並べ替え
        
        # 刺激時のデータを抽出
        stim_all_df = stimuli_exp(data_df)
        
        # 統計量をタスクごとに計算
        statistics_df = stat_calc(stim_all_df, i_sub)
        
        # ここで，いらないタスクを削除する．
        del_list = DEL_LIST[i_sub] # 削除するタスク番号リストを取り出す
        for index, item in enumerate(del_list):
            statistics_df = statistics_df[statistics_df["Task"]!=item] # 対象タスクのデータを削除
    
        # 各指標毎に，標準化する
        s_data_df = standardize(statistics_df)
        
        # データを指標毎に分けて取り出す
        mean_df = s_data_df[s_data_df.Statistics=="Mean"]
        max_df = s_data_df[s_data_df.Statistics=="Max"]
        min_df = s_data_df[s_data_df.Statistics=="Min"]
        std_df = s_data_df[s_data_df.Statistics=="Std"]
        
        # Excelファイルとして書き出す
        os.chdir(DATAPATH2)
        with pd.ExcelWriter(FILENAME_LIST2[i_sub]) as writer:
            mean_df.to_excel(writer, sheet_name="mean")
            max_df.to_excel(writer, sheet_name="max")
            min_df.to_excel(writer, sheet_name="min")
            std_df.to_excel(writer, sheet_name="std")
        
    
# 【関数定義】 ##################################################################
# データフレーム構造のデータを引数とする場合の微分関数(dがデータフレーム)
def diff_df(df, t):
    diffdf = pd.DataFrame(index=df.index, columns=[]) # 出力するデータフレームの作成
    for j in range(len(df.columns)):
        d = df.iloc[:, [j]] # j列目だけを選択(データフレーム型)
        d = d.values # Numpy配列に変換
        diffd = np.empty([1])
        diffd[0] = 0 # 先頭要素はゼロ
        for i in range(len(d)-1):
            x = (d[i+1]-d[i])/(t[i+1]-t[i])
            diffd = np.append(diffd, x)
        if j == 0:
            diffdf["η"] = diffd
        elif j == 1:
            diffdf["β"] = diffd
        elif j == 2:
            diffdf["HR"] = diffd
        elif j == 3:
            diffdf["LF/HF"] = diffd
        elif j == 4:
            diffdf["HF"] = diffd
        elif j == 5:
            diffdf["LF"] = diffd
    return diffdf

# 刺激提示時のデータを抽出する関数(引数はイベント情報と生体情報を含むデータフレーム)
def stimuli_exp(df):
    # 刺激を提示している部分を抽出(データフレームのまま)
    stim = pd.DataFrame(columns=COLUMN_TEMP1) # 各タスク刺激提示時一時保存df
    stim_all_df = pd.DataFrame(columns=COLUMN_TEMP1) # 全タスク刺激提示時保存df
    i_task = 1 # タスク数記録変数
    
    for i in range(len(df)): # 刺激区間を探し出すループ
        if df.iloc[i, 0] >= 2: #　フラグが立っている(第0列がEvent)
            stim = stim.append(df.iloc[i, :]) # フラグが経っている行を追加
        if len(stim) != 0 and df.iloc[i, 0] < 2: # フラグが消えたら
            stim = stim.assign(Task=i_task)
            i_task += 1 # タスク数の記録用
            stim_all_df = stim_all_df.append(stim, sort=False) # 刺激提示区間をパネルに格納
            stim = pd.DataFrame(columns=COLUMN_TEMP1) # stimを空に
    return stim_all_df

# 統計量を計算する関数（引数はタスクと計算対象が含まれるデータフレーム, 被験者番号）
def stat_calc(df, sub):
    # 抽出したデータの4指標を計算(回りくどいやり方かも)
    stat_all = pd.DataFrame(columns=COLUMN_TEMP2) # 全結果格納用(タスク順)
    mean_all = pd.DataFrame(columns=stat_all.columns) # 平均値格納用
    max_all = pd.DataFrame(columns=stat_all.columns) # 最大値格納用
    min_all = pd.DataFrame(columns=stat_all.columns) # 最小値格納用
    std_all = pd.DataFrame(columns=stat_all.columns) # 標準偏差値格納用
    
    for i_task in range(1, SESSION_LIST[sub]*TASK_LIST[sub]+1): # 全タスク分を計算
        # データ整理
        stat = pd.DataFrame(columns=df.columns) # statistics(統計量)：計算対象のデータフレーム
        stat = df[df.Task == i_task] # i_taskタスク目を抽出
        stat = stat.loc[:, COLUMN_TEMP2] # 計算対象のみにする（ならべかえ）
        
        # 平均値
        tmp = pd.DataFrame(columns=stat.columns) # データ格納用
        x = [] # 計算用
        for i in range(len(stat.columns)):
            x.append(stat.iloc[:, i].mean()) # 各生体情報の平均値
        x = pd.DataFrame(x, index=stat.columns, columns=["Mean"]).T # データフレームに変換(行と列の関係でややこしくなってます)
        tmp = tmp.append(x) # 結果をデータフレームに追加
        tmp = tmp.assign(Task=i_task)
        mean_all = mean_all.append(tmp, sort=False)
        stat_all = stat_all.append(tmp, sort=False)
        
        # 最大値
        tmp = pd.DataFrame(columns=stat.columns) # データ格納用
        x = [] # 計算用
        for i in range(len(stat.columns)):
            x.append(stat.iloc[:, i].max()) # 各生体情報の最大値
        x = pd.DataFrame(x, index=stat.columns, columns=["Max"]).T # データフレームに変換(行と列の関係でややこしくなってます)
        tmp = tmp.append(x) # 結果をデータフレームに追加
        tmp = tmp.assign(Task=i_task)
        max_all = max_all.append(tmp, sort=False)
        stat_all = stat_all.append(tmp, sort=False)
        
        # 最小値
        tmp = pd.DataFrame(columns=stat.columns) # データ格納用
        x = [] # 計算用
        for i in range(len(stat.columns)):
            x.append(stat.iloc[:, i].min()) # 各生体情報の最小値
        x = pd.DataFrame(x, index=stat.columns, columns=["Min"]).T # データフレームに変換(行と列の関係でややこしくなってます)
        tmp = tmp.append(x) # 結果をデータフレームに追加
        tmp = tmp.assign(Task=i_task)
        min_all = min_all.append(tmp, sort=False)
        stat_all = stat_all.append(tmp, sort=False)
        
        # 標準偏差
        tmp = pd.DataFrame(columns=stat.columns) # データ格納用
        x = [] # 計算用
        for i in range(len(stat.columns)):
            x.append(stat.iloc[:, i].std()) # 各生体情報の標準偏差値
        x = pd.DataFrame(x, index=stat.columns, columns=["Std"]).T # データフレームに変換(行と列の関係でややこしくなってます)
        tmp = tmp.append(x) # 結果をデータフレームに追加
        tmp = tmp.assign(Task=i_task)
        std_all = std_all.append(tmp, sort=False)
        stat_all = stat_all.append(tmp, sort=False)

    return stat_all

# 列ごとに標準化する関数（引数はタスク数込みのデータフレーム）
def standardize(df):
    # 平均値
    mean = pd.DataFrame(columns=COLUMN_TEMP3) # 計算対象のデータフレーム
    mean = df.loc["Mean"] # 平均値データを抽出
    task = mean["Task"] # タスク番号を保管しておく
    mean = scipy.stats.zscore(mean, axis=0) # 標準化
    mean = pd.DataFrame(mean) # データフレーム型に変換
    mean.columns = COLUMN_TEMP3 # 列名を指定
    mean = mean.assign(Task=task.values.tolist()) # タスク番号を標準化前のもので上書き
    mean["Statistics"] = "Mean" # 統計量の種類をマーク
    mean = mean.set_index("Task") # Taskをインデックスにセット
    # 最大値
    maximum = pd.DataFrame(columns=COLUMN_TEMP3) # 計算対象のデータフレーム
    maximum = df.loc["Max"] # 最大値値データを抽出
    task = maximum["Task"] # タスク番号を保管しておく
    maximum = scipy.stats.zscore(maximum, axis=0) # 標準化
    maximum = pd.DataFrame(maximum) # データフレーム型に変換
    maximum.columns = COLUMN_TEMP3 # 列名を指定
    maximum = maximum.assign(Task=task.values.tolist()) # タスク番号を標準化前のもので上書き
    maximum["Statistics"] = "Max" # 統計量の種類をマーク
    maximum = maximum.set_index("Task") # Taskをインデックスにセット
    # 最小値
    minimum = pd.DataFrame(columns=COLUMN_TEMP3) # 計算対象のデータフレーム
    minimum = df.loc["Min"] # 最小値データを抽出
    task = minimum["Task"] # タスク番号を保管しておく
    minimum = scipy.stats.zscore(minimum, axis=0) # 標準化
    minimum = pd.DataFrame(minimum) # データフレーム型に変換
    minimum.columns = COLUMN_TEMP3 # 列名を指定
    minimum = minimum.assign(Task=task.values.tolist()) # タスク番号を標準化前のもので上書き
    minimum["Statistics"] = "Min" # 統計量の種類をマーク
    minimum = minimum.set_index("Task") # Taskをインデックスにセット
    # 標準偏差
    std = pd.DataFrame(columns=COLUMN_TEMP3) # 計算対象のデータフレーム
    std = df.loc["Std"] # 標準偏差データを抽出
    task = std["Task"] # タスク番号を保管しておく
    std = scipy.stats.zscore(std, axis=0) # 標準化
    std = pd.DataFrame(std) # データフレーム型に変換
    std.columns = COLUMN_TEMP3 # 列名を指定
    std = std.assign(Task=task.values.tolist()) # タスク番号を標準化前のもので上書き
    std["Statistics"] = "Std" # 統計量の種類をマーク
    std = std.set_index("Task") # Taskをインデックスにセット
    
    #stat = pd.DataFrame(columns=COLUMN_TEMP1)
    stat = pd.concat([mean, maximum, minimum, std], sort=False)
    return stat
    
# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()