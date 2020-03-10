# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

主成分分析したい．
データは6（生体情報）×2（各微分）×4（統計量）=48次元
2つの主成分軸を表示
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
FILENAME_A = "SubA_rslt2.xlsx"
FILENAME_B = "SubB_rslt2.xlsx"
FILENAME_C = "SubC_rslt2.xlsx"
FILENAME_D = "SubD_rslt2.xlsx"
FILENAME_E = "SubE_rslt2.xlsx"
FILENAME_F = "SubF_rslt2.xlsx"

# 主観評価データがあるパスとそのファイル名
DATAPATH2 = "C:\\Users\\A_lifePC\\Desktop\\ファイル読み込み用\\Unpleasant1"
FILENAME2_A = "SubA_question.xlsx"
FILENAME2_B = "SubB_question.xlsx"
FILENAME2_C = "SubC_question.xlsx"
FILENAME2_D = "SubD_question.xlsx"
FILENAME2_E = "SubE_question.xlsx"
FILENAME2_F = "SubF_question.xlsx"

# 【メイン処理】 ##################################################################
def main():
    #　データがあるパスに作業ディレクトリ変更
    os.chdir(DATAPATH)
    # データフレームにデータ格納
    df_A = pd.read_excel(FILENAME_A)
    df_B = pd.read_excel(FILENAME_B)
    df_C = pd.read_excel(FILENAME_C)
    df_D = pd.read_excel(FILENAME_D)
    df_E = pd.read_excel(FILENAME_E)
    df_F = pd.read_excel(FILENAME_F)

    # まずは，データをNumpy配列に
    # 配列の各要素について，[eta...LFdiffの平均, 最大，最小，標準偏差]の順に格納する
    data = arrange_data(df_A)
    data = np.vstack((data, arrange_data(df_B))) # 行を追加していく
    data = np.vstack((data, arrange_data(df_C)))
    data = np.vstack((data, arrange_data(df_D)))
    data = np.vstack((data, arrange_data(df_E)))
    data = np.vstack((data, arrange_data(df_F)))
   
    # ここで，dataには列数の次元を持つ行数分のデータが格納されている
    
    # dataを列ごとに標準化(標準化の標準化)
    stan_data = scipy.stats.zscore(data, axis=0)
    
    # 刺激の種類でデータを分けたい．
    # 刺激の種類が記された配列を作成．
    os.chdir(DATAPATH2) #　データがあるパスに作業ディレクトリ変更
    df2_A = pd.read_excel(FILENAME2_A) # データフレームにデータ格納
    df2_B = pd.read_excel(FILENAME2_B)
    df2_C = pd.read_excel(FILENAME2_C)
    df2_D = pd.read_excel(FILENAME2_D)
    df2_E = pd.read_excel(FILENAME2_E)
    df2_F = pd.read_excel(FILENAME2_F)
    
    odor = df2_A["Stimulation"].values.tolist()
    odor.extend(df2_B["Stimulation"].values.tolist())
    odor.extend(df2_C["Stimulation"].values.tolist())
    odor.extend(df2_D["Stimulation"].values.tolist())
    odor.extend(df2_E["Stimulation"].values.tolist())
    odor.extend(df2_F["Stimulation"].values.tolist())
    
    odor = np.reshape(odor, (72,1)) # 刺激の種類
    
    # PCAしてグラフをプロット
    pca_plot(stan_data, odor)
    
    
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

# 主成分分析して，2次元プロットする関数
def pca_plot(data, odor):
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
    plt.scatter(un[:, 0], un[:, 1], c=[0.4, 0.6, 0.9])
    plt.scatter(non[:, 0], non[:, 1], c=[0.5, 0.5, 0.5])
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    
    # 主成分の次元ごとの寄与率を出力する
    print(pca.explained_variance_ratio_)

    # グラフを表示する
    plt.show()
    
    
# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()