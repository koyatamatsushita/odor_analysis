# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

主成分分析したい．
データは6（生体情報）×2（各微分）×4（統計量）=48次元
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# 【前処理】 ####################################################################
# 生体情報データがあるパスとそのファイル名
DATAPATH = "C:\\Users\\A_lifePC\\Desktop\\ファイル書き出し用"
FILENAME = "SubA_rslt2.xlsx"

# 主観評価データがあるパスとそのファイル名
DATAPATH2 = "C:\\Users\\A_lifePC\\Desktop\\ファイル読み込み用\\Unpleasant1"
FILENAME2 = "SubA_question.xlsx"

 # 列ラベルテンプレ
COLUMN_TEMP1 = ['Event', 'Time', 'Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Etadiff', 'Betadiff', 'HRdiff', 'LF/HFdiff', 'HFdiff', 'LFdiff']
COLUMN_TEMP2 = ['Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Etadiff', 'Betadiff', 'HRdiff', 'LF/HFdiff', 'HFdiff', 'LFdiff']
COLUMN_TEMP3 = ['Eta', 'Beta', 'HR', 'LF/HF', 'HF', 'LF', 'Etadiff', 'Betadiff', 'HRdiff', 'LF/HFdiff', 'HFdiff', 'LFdiff', 'Task']

# 【メイン処理】 ##################################################################
def main():
    #　データがあるパスに作業ディレクトリ変更
    os.chdir(DATAPATH)
    # データフレームにデータ格納
    df = pd.read_excel(FILENAME)
    
    #print(df) # 確認用
    
    # まずは，データをNumpy配列に
    # 配列の各要素について，[eta...LFdiffの平均, 最大，最小，標準偏差]の順に格納する
    data = arrange_data(df)
    
    #print(data) # 確認用
    
    # ここで，dataには列数の次元を持つ行数分のデータが格納されている
    
    # 刺激の種類でデータを分けたい．
    # 刺激の種類が記された配列を作成．
    
    os.chdir(DATAPATH2) #　データがあるパスに作業ディレクトリ変更
    df2 = pd.read_excel(FILENAME2) # データフレームにデータ格納
    
    odor = df2["Stimulation"].values.tolist()
    odor = np.reshape(odor, (12,1))
    
    #print(odor)
    
    # 主成分分析する
    pca = PCA()
    pca.fit(data)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(data)
    
    # 匂い情報を追加
    transformed_odor = np.hstack((transformed, odor))
    
    # 匂い別に分ける
    un = transformed[np.any(transformed_odor=="Unpleasant", axis=1)]
    non = transformed[np.any(transformed_odor=="Nonodor", axis=1)]
    
    # グラフ描画サイズを設定する
    plt.figure(figsize=(8, 6))
    # 主成分をプロットする
    #plt.scatter(transformed[:, 0], transformed[:, 1])
    plt.scatter(un[:, 0], un[:, 1], c=[0.4, 0.6, 0.9])
    plt.scatter(non[:, 0], non[:, 1], c=[0.5, 0.5, 0.5])
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    
    # 主成分の次元ごとの寄与率を出力する
    print(pca.explained_variance_ratio_)

    # グラフを表示する
    plt.show()
    
    
# 【関数定義】 ##################################################################
# データをnumpy配列として整理する関数
def arrange_data(df):
    # まずは，データをNumpy配列に
    # 配列の各要素について，[eta...LFdiffの平均, 最大，最小，標準偏差]の順に格納する
    
    # データを格納する
    df_mean = df[df["Statistics"]=="Mean"].drop("Statistics", axis=1)
    df_max = df[df["Statistics"]=="Max"].drop("Statistics", axis=1)
    df_min = df[df["Statistics"]=="Min"].drop("Statistics", axis=1)
    df_std = df[df["Statistics"]=="Std"].drop("Statistics", axis=1)
    
    # 配列にするデータフレームの作成
    data_df = pd.concat([df_mean, df_max, df_min, df_std], axis=1, sort=False)
    # 配列に変換
    data = data_df.values
    
    return data
    
# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()