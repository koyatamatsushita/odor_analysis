# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

excelから実験結果データ読み取り
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd

# 【前処理】 ####################################################################
# データがあるパスとそのファイル名
DATAPATH = "C:\\Users\\A_lifePC\\Desktop\\ファイル読み込み用\\Unpleasant1"
FILENAME = "SubA.xlsx"

# Excelファイルとして書き出すパスとファイル名
DATAPATH2 = "C:\\Users\\A_lifePC\\Desktop\\ファイル書き出し用"
FILENAME2 = "SubA_rslt.xlsx"

# 【メイン処理】 ##################################################################
def main():
    #　データがあるパスに作業ディレクトリ変更
    os.chdir(DATAPATH)
    # データフレームにデータ格納
    data = pd.read_excel(FILENAME)
    
    #print(data) # 確認用
    
    """メモ書き(全てprintで表示するようにしています)
    
    # 列名を取得する
    print(data.columns)
    
    # インデックス名を取得する
    print(data.index)
    
    # 行数の確認
    print(len(data))
    
    # 次元数の確認
    print(data.shape)
    
    # カラム情報の一覧（カラム名とその型の一覧）
    print(data.info())
    
    # 各列の基礎統計量の確認
    # Rでいうところのsummary()
    print(data.describe()) # 平均、分散、4分位など
    
    # head / tail
    print(data.head(10)) #先頭10行を確認
    print(data.tail(10)) #先頭10行を確認
    
    """
    # 時間データを取り出す
    t = data["Time"]
    # 微分する対象のデータ列を取り出す
    #d = data["β"]
    df = pd.DataFrame({"η": data["η"],
                      "β": data["β"],
                      "HR": data["HR"],
                      "LF/HF": data["L/H30"],
                      "HF": data["HF30"],
                      "LF": data["LF30"],})
  
    #print(df) #確認用
    #print(pd.concat([df, data["η"]], axis=1)) #確認用
    #print(range(len(d)-1)) #確認用
    
    # differential関数で微分
    diffdf = diff_df(df, t)
    
    print(diffdf) # 確認用
    
    # Excelファイルとして書き出すデータフレームの作成
    df_exp = pd.DataFrame({"Time": t,
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
    
    print(df_exp) # 確認用
    
    # Excelファイルとして書き出す
    os.chdir(DATAPATH2)
    df_exp.to_excel(FILENAME2)

# 【関数定義】 ##################################################################
# リスト構造のデータを引数とする場合の微分関数(dがリスト)
def diff_list(d, t):
    diffd = [0] # 先頭要素はゼロ
    for i in range(len(d)-1):
        x = (d[i+1]-d[i])/(t[i+1]-t[i])
        diffd.append(x)
    return diffd 

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

# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()
