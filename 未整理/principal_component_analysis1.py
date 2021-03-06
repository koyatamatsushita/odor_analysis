# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

主観評価と生体情報をそれぞれ主成分分析．
生体情報は全指標としている．
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 【前処理】 ####################################################################
# 生体情報データがあるパスとそのファイル名
DATAPATH = "C:\\Users\\A_lifePC\\Desktop\\ファイル書き出し用"
FILENAME_LIST = ["SubA_standardize.xlsx",
                 "SubB_standardize.xlsx",
                 "SubC_standardize.xlsx",
                 "SubD_standardize.xlsx",
                 "SubE_standardize.xlsx",
                 "SubF_standardize.xlsx"]

# 主観評価データがあるパスとそのファイル名
DATAPATH2 = "C:\\Users\\A_lifePC\\Desktop\\ファイル読み込み用\\Unpleasant1"
FILENAME_LIST2 = ["SubA_question.xlsx",
                  "SubB_question.xlsx",
                  "SubC_question.xlsx",
                  "SubD_question.xlsx",
                  "SubE_question.xlsx",
                  "SubF_question.xlsx"]

# 削除する不要なタスクのタスク番号リスト
DEL_LIST = [[10, 11, 12],
            [10, 11, 12],
            [1, 2, 3, 10, 11, 12],
            [10, 11, 12],
            [10, 11, 12],
            [10, 11, 12]]

# 【メイン処理】 ##################################################################
def main():
    # まずは生体情報のデータstan_dataを取り出す -------------------------------------
    
    #　データがあるパスに作業ディレクトリ変更
    os.chdir(DATAPATH)
    # data格納用のデータフレームを準備
    data_df = pd.DataFrame([])
    
    for i_sub in range(len(FILENAME_LIST)):
        # データフレームにデータ格納(このデータはすでに標準化済み)
        mean_df = pd.read_excel(FILENAME_LIST[i_sub], sheet_name="mean").drop("Statistics", axis=1)
        max_df = pd.read_excel(FILENAME_LIST[i_sub], sheet_name="max").drop("Statistics", axis=1)
        min_df = pd.read_excel(FILENAME_LIST[i_sub], sheet_name="min").drop("Statistics", axis=1)
        std_df = pd.read_excel(FILENAME_LIST[i_sub], sheet_name="std").drop("Statistics", axis=1)
        # [平均，最大，最小，標準偏差]の順に横に並べる
        df = pd.concat([mean_df, max_df.drop("Task", axis=1), min_df.drop("Task", axis=1), std_df.drop("Task", axis=1)], axis=1, sort=False)
        # 各被験者の結果を縦に連結(ついでに標準化する)
        data_df = data_df.append(df)
    
    # タスク番号の削除
    data2_df = data_df.drop(["Task"], axis=1)
    
    # dataを列ごとに標準化(標準化の標準化)
    stan_data = scipy.stats.zscore(data2_df, axis=0)
    # -------------------------------------------------------------------------
    
    # 次に主観評価のデータq_stan_dataを取り出す ------------------------------------
    os.chdir(DATAPATH2) #　データがあるパスに作業ディレクトリ変更
    # data格納用のデータフレームを準備
    q_data_df = pd.DataFrame([])
    for i_sub in range(len(FILENAME_LIST2)):
        # データフレームにデータ格納
        q_df = pd.read_excel(FILENAME_LIST2[i_sub])
        # 各被験者の結果を縦に連結(ついでに標準化する)
        q_data_df = q_data_df.append(arrange_data(q_df, i_sub))
    
    # タスク番号, 刺激の種類の削除
    q_data2_df = q_data_df.drop(["No", "Stimulation"], axis=1)
    
    # dataを列ごとに標準化(標準化の標準化)
    q_stan_data = scipy.stats.zscore(q_data2_df, axis=0)
    # -------------------------------------------------------------------------
    
    # 刺激の種類のndarray配列を用意
    odor = q_data_df["Stimulation"].values.tolist()
    odor = np.reshape(odor, (len(odor),1)) # 刺激の種類
    
    # PCAして，各第2主成分までを平面プロット
    pca_2Dplot2(q_stan_data, stan_data, odor)
    
    # PCAして，主観評価の第1主成分と生体情報の第2主成分までを3Dプロット
    pca_3Dplot(q_stan_data, stan_data, odor)
    
    
# 【関数定義】 ##################################################################
# データを整理する関数
def arrange_data(df, i_sub):
    
    # 不要なタスクを削除する
    del_list = DEL_LIST[i_sub] # 削除するタスク番号リストを取り出す
    for index, item in enumerate(del_list):
        df = df[df["No"]!=item] # 対象タスクのデータを削除
    
    task = df["No"] # タスク番号を保管しておく
    stimulation = df["Stimulation"] # 刺激の種類を保管しておく
    
    # 刺激の種類，匂い強度削除
    df = df.drop(["Stimulation","Intensity"], axis=1)
    columns = df.columns
    
    # 列ごとに標準化する
    data = scipy.stats.zscore(df, axis=0)
    # データフレーム型に戻す
    df = pd.DataFrame(data)
    df.columns = columns # カラム名も戻す
    
    df = df.assign(No=task.values.tolist()) # タスク番号を標準化前のもので上書き
    df = df.assign(Stimulation=stimulation.values.tolist()) # 刺激の種類を付け足す
    
    return df

# 主成分分析して，2次元プロットする関数
def pca_2Dplot(data1, data2, odor): # data1が主観評価，data2が生体情報
    
    # 主観評価の主成分分析する---------------------------------------------------
    pca1 = PCA()
    pca1.fit(data1)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed1 = pca1.fit_transform(data1)
    
    # 匂い情報を追加
    transformed1_odor = np.hstack((transformed1, odor))
    
    # 匂い別に分ける
    un1 = transformed1[np.any(transformed1_odor=="Unpleasant", axis=1)]
    non1 = transformed1[np.any(transformed1_odor=="Nonodor", axis=1)]
    
    # 主成分の次元ごとの寄与率を出力する
    print("主観評価の寄与率:")
    print(pca1.explained_variance_ratio_)
    
    # 生体情報の主成分分析する---------------------------------------------------
    pca2 = PCA()
    pca2.fit(data2)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed2 = pca2.fit_transform(data2)
    
    # 匂い情報を追加
    transformed2_odor = np.hstack((transformed2, odor))
    
    # 匂い別に分ける
    un2 = transformed2[np.any(transformed2_odor=="Unpleasant", axis=1)]
    non2 = transformed2[np.any(transformed2_odor=="Nonodor", axis=1)]
    
        # 主成分の次元ごとの寄与率を出力する
    print("生体情報の寄与率:")
    print(pca2.explained_variance_ratio_)
    
    # 主観評価，生体情報の各第2主成分までをプロットする--------------------------------
    # グラフ描画サイズを設定する
    plt.figure(figsize=(10, 5))
    # 1つ目のグラフ
    plt.subplot(1, 2, 1)
    # 主成分をプロットする
    plt.scatter(un1[:, 0], un1[:, 1], s=80, c=[0.4, 0.6, 0.9], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.scatter(non1[:, 0], non1[:, 1], s=80, c=[0.5, 0.5, 0.5], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.title('Subjective evaluation', fontsize=18)
    plt.xlabel('pc1', fontsize=18)
    plt.ylabel('pc2', fontsize=18)
    
    # 2つ目のグラフ
    plt.subplot(1, 2, 2)
    # 主成分をプロットする
    plt.scatter(un2[:, 0], un2[:, 1], s=80, c=[0.4, 0.6, 0.9], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.scatter(non2[:, 0], non2[:, 1], s=80, c=[0.5, 0.5, 0.5], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.title('Biometric information', fontsize=18)
    plt.xlabel('pc1', fontsize=18)
    plt.ylabel('pc2', fontsize=18)
    
    # グラフを表示する
    plt.tight_layout()  # タイトルの被りを防ぐ
    plt.show()

# 主成分分析して，2次元プロットする関数
def pca_2Dplot2(data1, data2, odor): # data1が主観評価，data2が生体情報
    
    # 主観評価の主成分分析する---------------------------------------------------
    pca1 = PCA()
    pca1.fit(data1)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed1 = pca1.fit_transform(data1)
    
    # 匂い情報を追加
    transformed1_odor = np.hstack((transformed1, odor))
    
    # 匂い別に分ける
    un1 = transformed1[np.any(transformed1_odor=="Unpleasant", axis=1)]
    non1 = transformed1[np.any(transformed1_odor=="Nonodor", axis=1)]
    
    # 主成分の次元ごとの寄与率を出力する
    print("主観評価の寄与率:")
    print(pca1.explained_variance_ratio_)
    
    # 生体情報の主成分分析する---------------------------------------------------
    pca2 = PCA()
    pca2.fit(data2)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed2 = pca2.fit_transform(data2)
    
    # 匂い情報を追加
    transformed2_odor = np.hstack((transformed2, odor))
    
    # 匂い別に分ける
    un2 = transformed2[np.any(transformed2_odor=="Unpleasant", axis=1)]
    non2 = transformed2[np.any(transformed2_odor=="Nonodor", axis=1)]
    
        # 主成分の次元ごとの寄与率を出力する
    print("生体情報の寄与率:")
    print(pca2.explained_variance_ratio_)
    
    # 主観評価，生体情報の各第2主成分までをプロットする--------------------------------
    # グラフ描画サイズを設定する
    plt.figure(figsize=(5, 5))
    
    # 主成分をプロットする
    plt.scatter(un1[:, 0], un2[:, 0], s=80, c=[0.4, 0.6, 0.9], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.scatter(non1[:, 0], non2[:, 0], s=80, c=[0.5, 0.5, 0.5], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.title('PCA scatter', fontsize=18)
    plt.xlabel('Subjective evaluation pc1', fontsize=18)
    plt.ylabel('Biometric information pc1', fontsize=18)
    
    # グラフを表示する
    plt.show()
    
    # -------------------------------------------------------------------------
    # 相関係数を計算
    # 計算用のデータフレームを作る
    cor_df = pd.DataFrame({"Sub": transformed1[:, 0], "Bio": transformed2[:, 0]})
    # 相関係数
    cor = cor_df.corr().iloc[0, 1]
    print("相関係数：")
    print(cor)

# 主成分分析して，3次元プロットする関数
def pca_3Dplot(data1, data2, odor): # data1が主観評価，data2が生体情報
    
    # 主観評価の主成分分析する---------------------------------------------------
    pca1 = PCA()
    pca1.fit(data1)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed1 = pca1.fit_transform(data1)
    
    # 匂い情報を追加
    transformed1_odor = np.hstack((transformed1, odor))
    
    # 匂い別に分ける
    un1 = transformed1[np.any(transformed1_odor=="Unpleasant", axis=1)]
    non1 = transformed1[np.any(transformed1_odor=="Nonodor", axis=1)]
    
    # 主成分の次元ごとの寄与率を出力する
    print("主観評価の寄与率:")
    print(pca1.explained_variance_ratio_)
    
    # 生体情報の主成分分析する---------------------------------------------------
    pca2 = PCA()
    pca2.fit(data2)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed2 = pca2.fit_transform(data2)
    
    # 匂い情報を追加
    transformed2_odor = np.hstack((transformed2, odor))
    
    # 匂い別に分ける
    un2 = transformed2[np.any(transformed2_odor=="Unpleasant", axis=1)]
    non2 = transformed2[np.any(transformed2_odor=="Nonodor", axis=1)]
    
        # 主成分の次元ごとの寄与率を出力する
    print("生体情報の寄与率:")
    print(pca2.explained_variance_ratio_)
    
    # 主観評価の第1主成分と生体情報の第2主成分までを3Dプロット-------------------------
    # 三次元プロット（主観評価PC1，生体情報PC1&PC2）
    # グラフ作成
    fig = plt.figure()
    ax = Axes3D(fig)
    
    # 軸ラベルの設定
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("emotion")
    
    # 表示範囲の設定
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-5,5)
    
    # グラフ描画
    # ms:点の大きさ mew:点の枠線の太さ mec:枠線の色
    # 4列目はetamax, 0列目はbetamax, 8列目はemotion
    ax.plot(un2[:, 0], un2[:, 1], un1[: ,0],
            "o", c=[0.4, 0.6, 0.9], ms=6, mew=1, mec='0.0')
    ax.plot(non2[:, 0], non2[:, 1], non1[: ,0],
            "o", c=[0.5, 0.5, 0.5], ms=6, mew=1, mec='0.0')
    plt.show()
    
# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()