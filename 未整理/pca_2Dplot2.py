# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

主観評価のファイルを参照して，主成分分析．
2つの主成分軸を表示する
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# 【前処理】 ####################################################################
# 主観評価データがあるパスとそのファイル名
DATAPATH = "C:\\Users\\A_lifePC\\Desktop\\ファイル読み込み用\\Unpleasant1"
FILENAME_LIST = ["SubA_question.xlsx",
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
    #　データがあるパスに作業ディレクトリ変更
    os.chdir(DATAPATH)
    # data格納用のデータフレームを準備
    data_df = pd.DataFrame([])
    
    for i_sub in range(len(FILENAME_LIST)):
        # データフレームにデータ格納
        df = pd.read_excel(FILENAME_LIST[i_sub])
        # 各被験者の結果を縦に連結(ついでに標準化する)
        data_df = data_df.append(arrange_data(df, i_sub))
    
    # タスク番号, 刺激の種類の削除
    data2_df = data_df.drop(["No", "Stimulation"], axis=1)
    
    # dataを列ごとに標準化(標準化の標準化)
    stan_data = scipy.stats.zscore(data2_df, axis=0)
    
    # 刺激の種類のndarray配列を用意
    odor = data_df["Stimulation"].values.tolist()
    odor = np.reshape(odor, (len(odor),1)) # 刺激の種類
    
    # PCAしてグラフをプロット
    pca_plot(stan_data, odor)
    
    
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
    plt.title('Subjective evaluation')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    
    # 主成分の次元ごとの寄与率を出力する
    print(pca.explained_variance_ratio_)

    # グラフを表示する
    plt.show()
    
# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()