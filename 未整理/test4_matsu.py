# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

イベント情報より，各区間を判別してデータを切り分けたい．
イベントが0.15以上2未満なら，各区間の境目
イベントが2以上の区間が刺激提示区間

セッション毎に標準化して，各刺激時の4指標を抜き出したい．
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd

# 【前処理】 ####################################################################
# セッション数
SESSION = 4
# セッションごとのタスク数
TASK = 3
# 休憩の有無
REST = True

# データがあるパスとそのファイル名
DATAPATH = "C:\\Users\\A_lifePC\\Desktop\\ファイル書き出し用"
FILENAME = "SubA_rslt.xlsx"

# Excelファイルとして書き出すパスとファイル名
DATAPATH2 = "C:\\Users\\A_lifePC\\Desktop\\ファイル書き出し用"
FILENAME2 = "SubA_rslt2.xlsx"

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
                       "LF": data["LF"]})
    
    #print(df) #確認用
    
    event = df["Event"]
    
    #イベント箇所記録用のリストを作成
    count = []
    
    for i in range(len(event)):
        #刺激提示時以外のフラグを探す
        # イベントフラグが立っている フラグは大きすぎない 前後のフラグは立っていない
        if event[i] > 0.15 \
           and event[i] < 2 \
           and event[i-1] < 0.1 \
           and event[i+1] < 0.1: # （刺激提示時は前後どちらかのフラグは立っているはず）
            count.append(i)
        
    #print(count) # 確認用
    
    # この時点で，countにはフラグが経っている行数(0スタート)が格納されている．
    # セッションを抽出する(要改善：くそプログラム)
    if REST == True:
        session1 = df[count[0]:count[4]]
        session2 = df[count[4]:count[8]]
        session3 = df[count[9]:count[13]]
        session4 = df[count[13]:count[17]]
    else:
        session1 = df[count[0]:count[4]]
        session2 = df[count[4]:count[8]]
        session3 = df[count[8]:count[12]]
        session4 = df[count[12]:count[16]]
    
    # セッションごとにデータを取り出したリスト
    #session = [session1, session2, session3, session4]
    
    #print(session) # 確認用
    
    
    # セッション毎に，生体情報を標準化する
    
    # セッション１
    # 標準化するデータのみに絞る
    x = session1.drop(["Time", "Event"], axis=1)
    
    # 生体情報を標準化
    x = (x - x.mean()) / x.std()
    
    # 標準化したデータを格納するデータフレームに格納
    standardize1 = pd.DataFrame(index=session1.index, columns=[])
    standardize1 = pd.concat([session1["Event"], session1["Time"], x], axis=1)
    
    """
    データフレームメモ
    
    pd.assign():列追加
    pd.append()：行追加
    
    df.iloc[行，列]：要素抽出
    """
    
    # 刺激を提示している部分を抽出(データフレームのまま)
    stim = pd.DataFrame(columns=standardize1.columns) # 各タスク刺激提示時一時保存df
    stim_all = pd.DataFrame(columns=standardize1.columns) # 全タスク刺激提示時保存panel
    i_task = 1 # タスク数記録変数
    
    for i in range(len(standardize1)): # 刺激区間を探し出すループ
        if standardize1.iloc[i, 0] >= 2: #　フラグが立っている(第0列がEvent)
            stim = stim.append(standardize1.iloc[i, :]) # フラグが経っている行を追加
        if len(stim) != 0 and session1.iloc[i, 0] < 2: # フラグが消えたら
            stim = stim.assign(Task=i_task)
            i_task += 1 # タスク数の記録用
            stim_all = stim_all.append(stim, sort=False) # 刺激提示区間をパネルに格納
            #print(stim) # 確認用
            stim = pd.DataFrame(columns=standardize1.columns) # stimを空に
    #print(stim_all)
    
    #print(stim) # 確認用
    
    COLUMN_TEMP1 = ['Event', 'Time', 'Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF'] # 列ラベルテンプレ
    COLUMN_TEMP2 = ['Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF'] # 列ラベルテンプレ
    COLUMN_TEMP3 = ['Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Task'] # 列ラベルテンプレ
    
    # 抽出したデータの4指標を計算(回りくどいやり方かも)
    
    stat_all = pd.DataFrame(columns=COLUMN_TEMP2) # 全結果格納用(タスク順)
    stat_all2 = pd.DataFrame(columns=COLUMN_TEMP2) # 全結果格納用（指標順）
    mean_all = pd.DataFrame(columns=stat_all.columns) # 平均値格納用
    max_all = pd.DataFrame(columns=stat_all.columns) # 最大値格納用
    min_all = pd.DataFrame(columns=stat_all.columns) # 最小値格納用
    std_all = pd.DataFrame(columns=stat_all.columns) # 標準偏差値格納用
    
    
    for i_task in range(1, TASK+1): # 1セッション3タスク分を計算
        # データ整理
        stat = pd.DataFrame(columns=stim_all.columns) # statistics(統計量)：計算対象のデータフレーム
        stat = stim_all[stim_all.Task == i_task] # i_taskタスク目を抽出
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
        
    stat_all2 = stat_all2.append([mean_all, max_all, min_all, std_all], sort=False)
    
    print(stat_all) # 確認用
    
    
    """
    # 刺激を提示している部分の時間を抽出(リストにして取り出す)
    t = session1["Time"].values.tolist() # 時間リスト
    s = session1["Event"].values.tolist() # イベントリスト
    stim = [] # 刺激提示時間一時保存リスト
    stim_time = [] # 刺激提示時間保存リスト
    for i in range(len(s)): # 刺激区間を探し出すループ
        if s[i] >= 2: #　フラグが立っている
            stim.append(t[i])
        if stim != [] and s[i] <2: # フラグが消えたら
            stim_time.append(stim) # 刺激提示区間をリストに格納
            stim = []
            
    print(stim_time) # 確認用
    """    
    
    """        
   　# 刺激提示時を抜き出す（配列がいいかな）
    stim_all = [] # 刺激提示時格納用
    for i in range(len(session1.columns)): # 生体情報の数だけ繰り返し
        for j in range(SESSION*TASK): # 刺激数繰り返し
            stim_time[j]
    """          
    
    """
    # セッション2
    # 標準化するデータのみに絞る
    x = session2.drop(["Time", "Event"], axis=1)
    
    # 生体情報を標準化
    x = (x - x.mean()) / x.std()
    
    # 標準化したデータを格納するデータフレームに格納
    standardize2 = pd.DataFrame(index=session2.index, columns=[])
    standardize2 = pd.concat([session2["Time"], session2["Event"], x], axis=1)
    
    # セッション3
    # 標準化するデータのみに絞る
    x = session3.drop(["Time", "Event"], axis=1)
    
    # 生体情報を標準化
    x = (x - x.mean()) / x.std()
    
    # 標準化したデータを格納するデータフレームに格納
    standardize3 = pd.DataFrame(index=session3.index, columns=[])
    standardize3 = pd.concat([session3["Time"], session3["Event"], x], axis=1)
    
    # セッション4
    # 標準化するデータのみに絞る
    x = session4.drop(["Time", "Event"], axis=1)
    
    # 生体情報を標準化
    x = (x - x.mean()) / x.std()
    
    # 標準化したデータを格納するデータフレームに格納
    standardize4 = pd.DataFrame(index=session4.index, columns=[])
    standardize4 = pd.concat([session4["Time"], session4["Event"], x], axis=1)
    
    #print(standardize1)
    #print(standardize2)
    #print(standardize3)
    #print(standardize4)
    
    #↓一つでもNULLがあると，全部NULLになってしまうようだ．
    #standardize = pd.concat([standardize1, standardize2, standardize3, standardize4], axis=1)
    
    # Excelファイルとして書き出す
    #os.chdir(DATAPATH2)
    #standardize.to_excel(FILENAME2)
    """
# 【関数定義】 ##################################################################
    
# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()
