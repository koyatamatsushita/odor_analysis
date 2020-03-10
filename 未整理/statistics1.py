# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

イベント情報より，刺激提示時の生体情報を取り出し，その時の
4指標（統計量）を計算して，標準化するプログラム
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats

# 【前処理】 ####################################################################
# セッション数
SESSION = 4
# セッションごとのタスク数
TASK = 3

# データがあるパスとそのファイル名
DATAPATH = "C:\\Users\\A_lifePC\\Desktop\\ファイル書き出し用"
FILENAME = "SubF_rslt.xlsx"

# Excelファイルとして書き出すパスとファイル名
DATAPATH2 = "C:\\Users\\A_lifePC\\Desktop\\ファイル書き出し用"
FILENAME2 = "SubF_rslt2.xlsx"

 # 列ラベルテンプレ
COLUMN_TEMP1 = ['Event', 'Time', 'Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Etadiff', 'Betadiff', 'HRdiff', 'LF/HFdiff', 'HFdiff', 'LFdiff']
COLUMN_TEMP2 = ['Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Etadiff', 'Betadiff', 'HRdiff', 'LF/HFdiff', 'HFdiff', 'LFdiff']
COLUMN_TEMP3 = ['Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Etadiff', 'Betadiff', 'HRdiff', 'LF/HFdiff', 'HFdiff', 'LFdiff', 'Task']
# 【メイン処理】 ##################################################################
def main():
    #　データがあるパスに作業ディレクトリ変更
    os.chdir(DATAPATH)
    # データフレームにデータ格納
    data = pd.read_excel(FILENAME)
    
    # 対象のデータ列を取り出す
    df = pd.DataFrame({"Event": data["Event"],
                       "Time": data["Time"],
                       "Eta": data["Eta"],
                       "Beta": data["Beta"],
                       "HR": data["HR"],
                       "LF/HF": data["LF/HF"],
                       "HF": data["HF"],
                       "LF": data["LF"],
                       "Etadiff": data["Etadiff"],
                       "Betadiff": data["Betadiff"],
                       "HRdiff": data["HRdiff"],
                       "LF/HFdiff": data["LF/HFdiff"],
                       "HFdiff": data["HFdiff"],
                       "LFdiff": data["LFdiff"]})
    
    # 刺激時のデータを抽出
    stim_all = stimuli_exp(df)

    # 統計量をタスクごとに計算
    statistics = stat_calc(stim_all)
    
    # 各指標毎に，標準化する
    s_data = standardize(statistics)
    
    # Excelファイルとして書き出す
    os.chdir(DATAPATH2)
    s_data.to_excel(FILENAME2)
    
# 【関数定義】 ##################################################################
# 刺激提示時のデータを抽出する関数(引数はイベント情報と生体情報を含むデータフレーム)
def stimuli_exp(df):
    # 刺激を提示している部分を抽出(データフレームのまま)
    stim = pd.DataFrame(columns=COLUMN_TEMP1) # 各タスク刺激提示時一時保存df
    stim_all = pd.DataFrame(columns=COLUMN_TEMP1) # 全タスク刺激提示時保存df
    i_task = 1 # タスク数記録変数
    
    for i in range(len(df)): # 刺激区間を探し出すループ
        if df.iloc[i, 0] >= 2: #　フラグが立っている(第0列がEvent)
            stim = stim.append(df.iloc[i, :]) # フラグが経っている行を追加
        if len(stim) != 0 and df.iloc[i, 0] < 2: # フラグが消えたら
            stim = stim.assign(Task=i_task)
            i_task += 1 # タスク数の記録用
            stim_all = stim_all.append(stim, sort=False) # 刺激提示区間をパネルに格納
            stim = pd.DataFrame(columns=COLUMN_TEMP1) # stimを空に
    return stim_all

# 統計量を計算する関数（引数はタスクと計算対象が含まれるデータフレーム）
def stat_calc(df):
    # 抽出したデータの4指標を計算(回りくどいやり方かも)
    stat_all = pd.DataFrame(columns=COLUMN_TEMP2) # 全結果格納用(タスク順)
    mean_all = pd.DataFrame(columns=stat_all.columns) # 平均値格納用
    max_all = pd.DataFrame(columns=stat_all.columns) # 最大値格納用
    min_all = pd.DataFrame(columns=stat_all.columns) # 最小値格納用
    std_all = pd.DataFrame(columns=stat_all.columns) # 標準偏差値格納用
    
    for i_task in range(1, SESSION*TASK+1): # 全タスク分を計算
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
    mean = pd.DataFrame(columns=COLUMN_TEMP1) # 計算対象のデータフレーム
    mean = df.loc["Mean"] # 平均値データを抽出
    mean = scipy.stats.zscore(mean) # 標準化
    mean = pd.DataFrame(mean) # データフレーム型に変換
    mean.columns = [COLUMN_TEMP3] # 列名を指定
    mean["Task"] = range(1, 13) # タスク番号を標準化前に戻す
    mean["Statistics"] = "Mean" # 統計量の種類をマーク
    mean = mean.set_index("Task") # Taskをインデックスにセット
    # 最大値
    maximum = pd.DataFrame(columns=COLUMN_TEMP1) # 計算対象のデータフレーム
    maximum = df.loc["Max"] # 最大値値データを抽出
    maximum = scipy.stats.zscore(maximum) # 標準化
    maximum = pd.DataFrame(maximum) # データフレーム型に変換
    maximum.columns = [COLUMN_TEMP3] # 列名を指定
    maximum["Task"] = range(1, 13) # タスク番号を標準化前に戻す
    maximum["Statistics"] = "Max" # 統計量の種類をマーク
    maximum = maximum.set_index("Task") # Taskをインデックスにセット
    # 最小値
    minimum = pd.DataFrame(columns=COLUMN_TEMP1) # 計算対象のデータフレーム
    minimum = df.loc["Min"] # 最小値データを抽出
    minimum = scipy.stats.zscore(minimum) # 標準化
    minimum = pd.DataFrame(minimum) # データフレーム型に変換
    minimum.columns = [COLUMN_TEMP3] # 列名を指定
    minimum["Task"] = range(1, 13) # タスク番号を標準化前に戻す
    minimum["Statistics"] = "Min" # 統計量の種類をマーク
    minimum = minimum.set_index("Task") # Taskをインデックスにセット
    # 標準偏差
    std = pd.DataFrame(columns=COLUMN_TEMP1) # 計算対象のデータフレーム
    std = df.loc["Std"] # 標準偏差データを抽出
    std = scipy.stats.zscore(std) # 標準化
    std = pd.DataFrame(std) # データフレーム型に変換
    std.columns = [COLUMN_TEMP3] # 列名を指定
    std["Task"] = range(1, 13) # タスク番号を標準化前に戻す
    std["Statistics"] = "Std" # 統計量の種類をマーク
    std = std.set_index("Task") # Taskをインデックスにセット
    
    stat = pd.DataFrame(columns=COLUMN_TEMP1)
    stat = pd.concat([mean, maximum, minimum, std])
    return stat
    
# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()
