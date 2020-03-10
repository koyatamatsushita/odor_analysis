# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

独立成分分析してみる．
主観評価と生体情報を分析しているので，まったく意味のないプログラム
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt
import matplotlib.cm as cm
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

COLUMNS48 = ["Eta_mean", "Beta_mean", "HR_mean", "LF/HF_mean", "HF_mean", "LF_mean",
             "Etadiff_mean", "Betadiff_mean", "HRdiff_mean", "LF/HFdiff_mean", "HFdiff_mean", "LFdiff_mean",
             "Eta_max", "Beta_max", "HR_max", "LF/HF_max", "HF_max", "LF_max",
             "Etadiff_max", "Betadiff_max", "HRdiff_max", "LF/HFdiff_max", "HFdiff_max", "LFdiff_max",
             "Eta_min", "Beta_min", "HR_min", "LF/HF_min", "HF_min", "LF_min",
             "Etadiff_min", "Betadiff_min", "HRdiff_min", "LF/HFdiff_min", "HFdiff_min", "LFdiff_min",
             "Eta_std", "Beta_std", "HR_std", "LF/HF_std", "HF_std", "LF_std",
             "Etadiff_std", "Betadiff_std", "HRdiff_std", "LF/HFdiff_std", "HFdiff_std", "LFdiff_std"]
# 【メイン処理】 ##################################################################
def main():
    # まずは生体情報のデータeta_data, beta_dataを取り出す -------------------------------------
    
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
    data2_df.columns = COLUMNS48
    
    # dataを列ごとに標準化(標準化の標準化)
    stan_data = scipy.stats.zscore(data2_df, axis=0)
    # データフレーム型に変換
    stan_data_df = pd.DataFrame(stan_data, columns=COLUMNS48)
    
    # 粘性，剛性をそれぞれ取り出す
    eta_data_df = stan_data_df.iloc[:, [0, 6, 12, 18, 24, 30, 36, 42]]
    beta_data_df = stan_data_df.iloc[:, [1, 7, 13, 19, 25, 31, 37, 43]]
    eta_data = eta_data_df.values
    beta_data = beta_data_df.values
    
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
    
    # 主観評価の主成分分析する---------------------------------------------------
    pca1 = PCA()
    pca1.fit(q_stan_data)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed1 = pca1.fit_transform(q_stan_data)
    # データフレーム型に変換
    transformed1 = pd.DataFrame(transformed1, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", ])
    
    # -------------------------------------------------------------------------
    
    # 相関係数を計算
    # 計算用のデータフレームを作る
    cor_df = pd.concat([transformed1["PC1"], stan_data_df], axis=1, sort=False)
    # 相関データフレーム
    cor_df = cor_df.corr()["PC1"]
    
    # 相関が0.45以上の指標だけをつかって主成分分析------------------------------------
    # 相関が0.45以上の指標を抜き出す
    new_stan_data_df = stan_data_df.drop(cor_df[cor_df<0.45].index, axis=1)
    new_stan_data = new_stan_data_df.values
    
    # PCAする(返り値はndarray)
    un_score, non_score = mypca(q_stan_data, new_stan_data, odor)
    my2Dplot(un_score, non_score)

    # ICAする
    # データの準備
    ica_data = np.vstack((un_score, non_score))
    # FastICAで独立成分分析
    ica = FastICA()
    ica.fit(ica_data)
    Uica = ica.components_.T
    Aica = ica.transform(ica_data).T
    
    Uica = Uica / np.sqrt((Uica**2).sum(axis=0))
    
    un = Aica[:, 0:len(un_score)]
    non = Aica[:, len(un_score):]
    # グラフ描画
    plt.figure(figsize=(5, 5))
    plt.scatter(un[0], un[1], s=80, c=[0.4, 0.6, 0.9], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.scatter(non[0], non[1], s=80, c=[0.5, 0.5, 0.5], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.title("ICA scatter", fontsize=18)
    
    # グラフを表示する
    plt.tight_layout()  # タイトルの被りを防ぐ
    plt.show()
    
    """
    # 散布図をプロット
    my2Dplot(un_score, non_score)
    my3Dplot(un_score, non_score)
    """
    
# 【関数定義】 ##################################################################
# データを整理する関数　\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
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

# 与えられたデータをそれぞれPC1まで主成分分析して，匂い刺激別に分ける関数 \\\\\\\\\\\\\\\\\\
def mypca(data1, data2, odor):
    # 主観評価の主成分分析する---------------------------------------------------
    pca1 = PCA(n_components=1) # 第1主成分だけを算出する
    pca1.fit(data1)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed1 = pca1.fit_transform(data1)
    
    # 匂い情報を追加
    transformed1_odor = np.hstack((transformed1, odor))
    
    # 匂い別に分ける
    un1 = transformed1[np.any(transformed1_odor=="Unpleasant", axis=1)]
    non1 = transformed1[np.any(transformed1_odor=="Nonodor", axis=1)]
    
    # 主成分の次元ごとの寄与率を出力する
    print("主観評価の寄与率:\n", pca1.explained_variance_ratio_, "\n")
    
    # 生体情報の主成分分析する---------------------------------------------------
    pca2 = PCA(n_components=1) # 第1主成分だけを算出する
    pca2.fit(data2)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed2 = pca2.fit_transform(data2)
    
    # 匂い情報を追加
    transformed2_odor = np.hstack((transformed2, odor))
    
    # 匂い別に分ける
    un2 = transformed2[np.any(transformed2_odor=="Unpleasant", axis=1)]
    non2 = transformed2[np.any(transformed2_odor=="Nonodor", axis=1)]
    
    # 主成分の次元ごとの寄与率を出力する
    print("生体情報の寄与率:\n", pca2.explained_variance_ratio_, "\n")
    
    # 相関係数を計算 -----------------------------------------------------------
    # 主観評価と血管粘性
    # 計算用のデータフレームを作る
    cor_df = pd.DataFrame(np.hstack((transformed1, transformed2)), columns=["Sub", "Bio"])
    # 相関データフレーム
    cor_df = cor_df.corr()
    # 相関係数を抽出
    cor = cor_df.iloc[0, 1]
    print("主観評価と生体情報の相関係数：\n", cor, "\n")

    # データをまとめる(返り値用)
    # ndarray行列[主観評価スコア，生体情報スコア]
    un_score = np.hstack((un1, un2))
    non_score = np.hstack((non1, non2))

    return un_score, non_score

# 与えられたデータをそれぞれPC1まで主成分分析して，匂い刺激別に分ける関数 \\\\\\\\\\\\\\\\\\
def mypca2(data1, data2, data3, odor):
    # 主観評価の主成分分析する---------------------------------------------------
    pca1 = PCA(n_components=1) # 第1主成分だけを算出する
    pca1.fit(data1)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed1 = pca1.fit_transform(data1)
    
    # 匂い情報を追加
    transformed1_odor = np.hstack((transformed1, odor))
    
    # 匂い別に分ける
    un1 = transformed1[np.any(transformed1_odor=="Unpleasant", axis=1)]
    non1 = transformed1[np.any(transformed1_odor=="Nonodor", axis=1)]
    
    # 主成分の次元ごとの寄与率を出力する
    print("主観評価の寄与率:\n", pca1.explained_variance_ratio_, "\n")
    
    # 生体情報の主成分分析する---------------------------------------------------
    pca2 = PCA(n_components=1) # 第1主成分だけを算出する
    pca2.fit(data2)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed2 = pca2.fit_transform(data2)
    
    # 匂い情報を追加
    transformed2_odor = np.hstack((transformed2, odor))
    
    # 匂い別に分ける
    un2 = transformed2[np.any(transformed2_odor=="Unpleasant", axis=1)]
    non2 = transformed2[np.any(transformed2_odor=="Nonodor", axis=1)]
    
    # 主成分の次元ごとの寄与率を出力する
    print("血管粘性の寄与率:\n", pca2.explained_variance_ratio_, "\n")
    
    
    pca3 = PCA(n_components=1) # 第1主成分だけを算出する
    pca3.fit(data3)
    
    # 分析結果を元にデータセットを主成分に変換する
    transformed3 = pca3.fit_transform(data3)
    
    # 匂い情報を追加
    transformed3_odor = np.hstack((transformed3, odor))
    
    # 匂い別に分ける
    un3 = transformed3[np.any(transformed3_odor=="Unpleasant", axis=1)]
    non3 = transformed3[np.any(transformed3_odor=="Nonodor", axis=1)]
    
    # 主成分の次元ごとの寄与率を出力する
    print("血管剛性の寄与率:\n", pca3.explained_variance_ratio_, "\n")
    
    # 相関係数を計算 -----------------------------------------------------------
    # 主観評価と血管粘性
    # 計算用のデータフレームを作る
    cor_df = pd.DataFrame(np.hstack((transformed1, transformed2)), columns=["Sub", "Eta"])
    # 相関データフレーム
    cor_df = cor_df.corr()
    # 相関係数を抽出
    cor = cor_df.iloc[0, 1]
    print("主観評価と血管粘性の相関係数：\n", cor, "\n")

    # 主観評価と血管剛性    
    # 計算用のデータフレームを作る
    cor_df = pd.DataFrame(np.hstack((transformed1, transformed3)), columns=["Sub", "Beta"])
    # 相関データフレーム
    cor_df = cor_df.corr()
    # 相関係数を抽出
    cor = cor_df.iloc[0, 1]
    print("主観評価と血管剛性の相関係数：\n", cor, "\n")
    
    # 血管粘性と血管剛性
    # 計算用のデータフレームを作る
    cor_df = pd.DataFrame(np.hstack((transformed2, transformed3)), columns=["Eta", "Beta"])
    # 相関データフレーム
    cor_df = cor_df.corr()
    # 相関係数を抽出
    cor = cor_df.iloc[0, 1]
    print("血管粘性と血管剛性の相関係数：\n", cor, "\n")
    
    # データをまとめる(返り値用)
    # ndarray行列[主観評価スコア，生体情報スコア]
    un_score = np.hstack((un1, un2, un3))
    non_score = np.hstack((non1, non2, non3))

    return un_score, non_score

# 2次元プロットする関数(データ1，データ2，グラフタイトル，x軸ラベル，y軸ラベル) \\\\\\\\\\\\\\\\\
def my2Dplot(data1, data2):
    
    # グラフ描画サイズを設定する
    plt.figure(figsize=(5, 5))

    # 主成分をプロットする
    plt.scatter(data1[:, 0], data1[:, 1], s=80, c=[0.4, 0.6, 0.9], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.scatter(data2[:, 0], data2[:, 1], s=80, c=[0.5, 0.5, 0.5], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.title("PCA scatter", fontsize=18)
    plt.xlabel("Subjective evaluation pc1", fontsize=18)
    plt.ylabel("Biometric information pc1", fontsize=18)
    
    # グラフを表示する
    plt.tight_layout()  # タイトルの被りを防ぐ
    plt.show()

# 3次元プロットする関数
def my3Dplot(data1, data2):
    
    # 三次元プロット（主観評価PC1，生体情報PC1&PC2）
    # グラフ作成
    fig = plt.figure()
    ax = Axes3D(fig)
    
    # 軸ラベルの設定
    ax.set_xlabel("Eta")
    ax.set_ylabel("Beta")
    ax.set_zlabel("Emotion")
    
    # 表示範囲の設定
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    
    # グラフ描画
    # ms:点の大きさ mew:点の枠線の太さ mec:枠線の色
    # 4列目はetamax, 0列目はbetamax, 8列目はemotion
    ax.plot(data1[:, 1], data1[:, 2], data1[: ,0],
            "o", c=[0.4, 0.6, 0.9], ms=6, mew=1, mec='0.0')
    ax.plot(data2[:, 1], data2[:, 2], data2[: ,0],
            "o", c=[0.5, 0.5, 0.5], ms=6, mew=1, mec='0.0')
    plt.show()
    """
    # 自動回転
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    """
# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()