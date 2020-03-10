# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 23:31:40 2018

@author: A_lifePC

まずは，「standardize1.py」を実行して，標準化したデータを用意しましょう．
そのデータを使って，まとめて解析できるプログラムにしたいなあと思います．
"F5"押したら諸々の解析結果をバーンて出すような
そんなプログラムにしたいなあ．
とりあえず，関数は考えて一般的に利用できるような形にしましょう．
どういった解析を行っているのかがぱっと見で分かるようにしたい，
かつ，解析の条件を変えやすいプログラムにしたいので，
データ整理は関数化して，main内では1行で済ませましょう．
"""

# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# 【前処理】 ####################################################################
# 生体情報データがあるパスとそのファイル名
DATAPATH = "C:\\Users\\A_lifePC\\Desktop\\ファイル書き出し用\\2018_11"
FILENAME_LIST = ["SubA_standardize.xlsx",
                 "SubB_standardize.xlsx",
                 "SubC_standardize.xlsx",
                 "SubD_standardize.xlsx",
                 "SubE_standardize.xlsx",
                 "SubF_standardize.xlsx"]

# 主観評価データがあるパスとそのファイル名
DATAPATH2 = "C:\\Users\\A_lifePC\\Desktop\\ファイル読み込み用\\2018_11\\Unpleasant1"
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
# 主観評価のカラム名のテンプレ
COLUMNS_Q = ["kassei", "wakuwaku", "kai", "rirakkusu", "hikassei", "taikutu", "hukai", "iraira"]

# 生体情報48指標のカラム名のテンプレ
COLUMNS48 = ["Eta_mean", "Beta_mean", "HR_mean", "LF/HF_mean", "HF_mean", "LF_mean",
             "Etadiff_mean", "Betadiff_mean", "HRdiff_mean", "LF/HFdiff_mean", "HFdiff_mean", "LFdiff_mean",
             "Eta_max", "Beta_max", "HR_max", "LF/HF_max", "HF_max", "LF_max",
             "Etadiff_max", "Betadiff_max", "HRdiff_max", "LF/HFdiff_max", "HFdiff_max", "LFdiff_max",
             "Eta_min", "Beta_min", "HR_min", "LF/HF_min", "HF_min", "LF_min",
             "Etadiff_min", "Betadiff_min", "HRdiff_min", "LF/HFdiff_min", "HFdiff_min", "LFdiff_min",
             "Eta_std", "Beta_std", "HR_std", "LF/HF_std", "HF_std", "LF_std",
             "Etadiff_std", "Betadiff_std", "HRdiff_std", "LF/HFdiff_std", "HFdiff_std", "LFdiff_std"]

# 【main処理】 ##################################################################
def main():
    # 解析に必要なデータの準備
    q_stan_data_df, stan_data_df, odor = find_mydata()
    
    print("主観評価をPCA------------------------------------------------------")
    q_transformed, q_un_df, q_non_df, q_loading = mypca1(q_stan_data_df.values, odor)
    print("主観評価PC1")
    myloading_top3(q_loading[0], COLUMNS_Q) # pc1の因子負荷量
    print("主観評価PC2")
    myloading_top3(q_loading[1], COLUMNS_Q) # pc2の因子負荷量
    print("生体情報をPCA------------------------------------------------------")
    transformed, un_df, non_df, loading = mypca1(stan_data_df.values, odor)
    print("生体情報PC1")
    myloading_top3(loading[0], COLUMNS48) # pc1の因子負荷量
    print("生体情報PC2")
    myloading_top3(loading[1], COLUMNS48) # pc2の因子負荷量
    
    
    ############ データ変数の準備 ############
    # 主観評価PC1
    q_score1 = np.hstack((q_un_df.iloc[:, 0].values, q_non_df.iloc[:, 0].values))
    q_un1 = q_un_df.iloc[:, 0].values
    q_non1 = q_non_df.iloc[:, 0].values
    # 主観評価PC2
    q_score2 = np.hstack((q_un_df.iloc[:, 1].values, q_non_df.iloc[:, 1].values))
    q_un2 = q_un_df.iloc[:, 1].values
    q_non2 = q_non_df.iloc[:, 1].values
    # 生体情報PC1
    score1 = np.hstack((un_df.iloc[:, 0].values, non_df.iloc[:, 0].values))
    un1 = un_df.iloc[:, 0].values
    non1 = non_df.iloc[:, 0].values
    # 生体情報PC2
    score2 = np.hstack((un_df.iloc[:, 1].values, non_df.iloc[:, 1].values))
    un2 = un_df.iloc[:, 1].values
    non2 = non_df.iloc[:, 1].values
    ########################################
    
    
    
    # t検定します
    print("主観評価PC1_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(q_un1, q_non1, equal_var=False), "\n")
    print("主観評価PC2_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(q_un2, q_non2, equal_var=False), "\n")
    print("生体情報PC1_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(un1, non1, equal_var=False), "\n")
    print("生体情報PC2_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(un2, non2, equal_var=False), "\n")

    
    print("主観評価PC1と主観評価PC2の相関を解析----------------------------------")
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, q_score2)
    # 描画する
    my2Dplot(np.vstack((q_un1, q_un2)).T, np.vstack((q_non1, q_non2)).T,
             "PCA scatter", "Subjective evaluation pc1", "Subjective evaluation pc2")
    
    print("生体情報PC1と生体情報PC2の相関を解析----------------------------------")
    # 相関係数，p値を計算する
    correlation_analysis(score1, score2)
    # 描画する
    my2Dplot(np.vstack((un1, un2)).T, np.vstack((non1, non2)).T,
             "PCA scatter", "Biometric information pc1", "Biometric information pc2")
    
    print("主観評価PC1と生体情報PC1の相関を解析----------------------------------")  
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, score1)
    # 描画する
    my2Dplot(np.vstack((q_un1, un1)).T, np.vstack((q_non1, non1)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information pc1")

    print("主観評価PC1と生体情報PC2の相関を解析----------------------------------")  
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, score2)
    # 描画する
    my2Dplot(np.vstack((q_un1, un2)).T, np.vstack((q_non1, non2)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information pc2")
    
    print("主観評価PC2と生体情報PC1の相関を解析----------------------------------")  
    # 相関係数，p値を計算する
    correlation_analysis(q_score2, score1)
    # 描画する
    my2Dplot(np.vstack((q_un2, un1)).T, np.vstack((q_non2, non1)).T,
             "PCA scatter", "Subjective evaluation pc2", "Biometric information pc1")

    print("主観評価PC2と生体情報PC2の相関を解析----------------------------------")  
    # 相関係数，p値を計算する
    correlation_analysis(q_score2, score2)
    # 描画する
    my2Dplot(np.vstack((q_un2, un2)).T, np.vstack((q_non2, non2)).T,
             "PCA scatter", "Subjective evaluation pc2", "Biometric information pc2")
    
    """
    # 末梢血管インデックスを削除したやつ\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    stan_data_df = stan_data_df.drop(["Eta_mean", "Beta_mean", "Etadiff_mean", "Betadiff_mean", "Eta_max", "Beta_max", "Etadiff_max", "Betadiff_max", 
                                          "Eta_min", "Beta_min", "Etadiff_min", "Betadiff_min", "Eta_std", "Beta_std", "Etadiff_std", "Betadiff_std"], axis=1)

    print("主観評価をPCA------------------------------------------------------")
    q_transformed, q_un_df, q_non_df, q_loading = mypca1(q_stan_data_df.values, odor)
    print("主観評価PC1")
    myloading_top3(q_loading[0], COLUMNS_Q) # pc1の因子負荷量
    print("主観評価PC2")
    myloading_top3(q_loading[1], COLUMNS_Q) # pc2の因子負荷量
    print("生体情報をPCA------------------------------------------------------")
    transformed, un_df, non_df, loading = mypca1(stan_data_df.values, odor)
    print("生体情報PC1")
    myloading_top3(loading[0], ["HR_mean", "LF/HF_mean", "HF_mean", "LF_mean",
                                "HRdiff_mean", "LF/HFdiff_mean", "HFdiff_mean", "LFdiff_mean",
                                "HR_max", "LF/HF_max", "HF_max", "LF_max",
                                "HRdiff_max", "LF/HFdiff_max", "HFdiff_max", "LFdiff_max",
                                "HR_min", "LF/HF_min", "HF_min", "LF_min",
                                "HRdiff_min", "LF/HFdiff_min", "HFdiff_min", "LFdiff_min",
                                "HR_std", "LF/HF_std", "HF_std", "LF_std",
                                "HRdiff_std", "LF/HFdiff_std", "HFdiff_std", "LFdiff_std"]) # pc1の因子負荷量
    print("生体情報PC2")
    myloading_top3(loading[1], ["HR_mean", "LF/HF_mean", "HF_mean", "LF_mean",
                                "HRdiff_mean", "LF/HFdiff_mean", "HFdiff_mean", "LFdiff_mean",
                                "HR_max", "LF/HF_max", "HF_max", "LF_max",
                                "HRdiff_max", "LF/HFdiff_max", "HFdiff_max", "LFdiff_max",
                                "HR_min", "LF/HF_min", "HF_min", "LF_min",
                                "HRdiff_min", "LF/HFdiff_min", "HFdiff_min", "LFdiff_min",
                                "HR_std", "LF/HF_std", "HF_std", "LF_std",
                                "HRdiff_std", "LF/HFdiff_std", "HFdiff_std", "LFdiff_std"]) # pc2の因子負荷量
    
    
    ############ データ変数の準備 ############
    # 主観評価PC1
    q_score1 = np.hstack((q_un_df.iloc[:, 0].values, q_non_df.iloc[:, 0].values))
    q_un1 = q_un_df.iloc[:, 0].values
    q_non1 = q_non_df.iloc[:, 0].values
    # 主観評価PC2
    q_score2 = np.hstack((q_un_df.iloc[:, 1].values, q_non_df.iloc[:, 1].values))
    q_un2 = q_un_df.iloc[:, 1].values
    q_non2 = q_non_df.iloc[:, 1].values
    # 生体情報PC1
    score1 = np.hstack((un_df.iloc[:, 0].values, non_df.iloc[:, 0].values))
    un1 = un_df.iloc[:, 0].values
    non1 = non_df.iloc[:, 0].values
    # 生体情報PC2
    score2 = np.hstack((un_df.iloc[:, 1].values, non_df.iloc[:, 1].values))
    un2 = un_df.iloc[:, 1].values
    non2 = non_df.iloc[:, 1].values
    ########################################
    
    
    
    # t検定します
    print("主観評価PC1_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(q_un1, q_non1, equal_var=False), "\n")
    print("主観評価PC2_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(q_un2, q_non2, equal_var=False), "\n")
    print("生体情報PC1_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(un1, non1, equal_var=False), "\n")
    print("生体情報PC2_不快と無臭のt検定結果：\n", sp.stats.ttest_ind(un2, non2, equal_var=False), "\n")

    
    print("主観評価PC1と主観評価PC2の相関を解析----------------------------------")
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, q_score2)
    # 描画する
    my2Dplot(np.vstack((q_un1, q_un2)).T, np.vstack((q_non1, q_non2)).T,
             "PCA scatter", "Subjective evaluation pc1", "Subjective evaluation pc2")
    
    print("生体情報PC1と生体情報PC2の相関を解析----------------------------------")
    # 相関係数，p値を計算する
    correlation_analysis(score1, score2)
    # 描画する
    my2Dplot(np.vstack((un1, un2)).T, np.vstack((non1, non2)).T,
             "PCA scatter", "Biometric information pc1", "Biometric information pc2")
    
    print("主観評価PC1と生体情報PC1の相関を解析----------------------------------")  
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, score1)
    # 描画する
    my2Dplot(np.vstack((q_un1, un1)).T, np.vstack((q_non1, non1)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information pc1")

    print("主観評価PC1と生体情報PC2の相関を解析----------------------------------")  
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, score2)
    # 描画する
    my2Dplot(np.vstack((q_un1, un2)).T, np.vstack((q_non1, non2)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information pc2")
    
    print("主観評価PC2と生体情報PC1の相関を解析----------------------------------")  
    # 相関係数，p値を計算する
    correlation_analysis(q_score2, score1)
    # 描画する
    my2Dplot(np.vstack((q_un2, un1)).T, np.vstack((q_non2, non1)).T,
             "PCA scatter", "Subjective evaluation pc2", "Biometric information pc1")

    print("主観評価PC2と生体情報PC2の相関を解析----------------------------------")  
    # 相関係数，p値を計算する
    correlation_analysis(q_score2, score2)
    # 描画する
    my2Dplot(np.vstack((q_un2, un2)).T, np.vstack((q_non2, non2)).T,
             "PCA scatter", "Subjective evaluation pc2", "Biometric information pc2")
    """
 
    """
    # 生体情報のPCAのいんしふかりょうが高い指標だけを使った解析############################
    # PC1の因子負荷量がの絶対値が0.5より小さい項目を削除
    loading_df = pd.DataFrame(loading[[0, 1]], index=["loading1", "loading2"], columns=COLUMNS48)
    pc1_stan_data_df = pd.concat([loading_df, stan_data_df], axis=0, sort=False)
    pc1_stan_data_df = pc1_stan_data_df.drop(pc1_stan_data_df.loc[:, np.fabs(pc1_stan_data_df.loc["loading1"])<0.5].columns, axis=1)
    pc1_stan_data_df = pc1_stan_data_df.drop(["loading1", "loading2"])
    print("因子負荷量が0.45以上の生体情報をPCA-----------------------------------")
    new_transformed, new_un_df, new_non_df, new_loading = mypca1(pc1_stan_data_df.values, odor)
  
    
    ############ データ変数の準備 ############
    # 因子負荷量の絶対値が0.5以上の生体情報PC1
    new_score1 = np.hstack((new_un_df.iloc[:, 0].values, new_non_df.iloc[:, 0].values))
    new_un1 = new_un_df.iloc[:, 0].values
    new_non1 = new_non_df.iloc[:, 0].values
    # 因子負荷量の絶対値が0.5以上の生体情報PC2
    new_score2 = np.hstack((new_un_df.iloc[:, 1].values, new_non_df.iloc[:, 1].values))
    new_un2 = new_un_df.iloc[:, 1].values
    new_non2 = new_non_df.iloc[:, 1].values
    ########################################
    
    
    print("因子負荷量の絶対値が0.5以上の生体情報PC1と生体情報PC2の相関を解析--------")
    # 相関係数，p値を計算する
    correlation_analysis(new_score1, new_score2)
    # 描画する
    my2Dplot(np.vstack((new_un1, new_un2)).T, np.vstack((new_non1, new_non2)).T,
             "PCA scatter", "Biometric information load>0.5 pc1", "Biometric information load>0.5 pc2")

    print("主観評価PC1と因子負荷量の絶対値が0.5以上の生体情報PC1の相関を解析--------")
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, new_score1)
    # 描画する
    my2Dplot(np.vstack((q_un1, new_un1)).T, np.vstack((q_non1, new_non1)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information load>0.5 pc1")

    print("主観評価PC1と因子負荷量の絶対値が0.5以上の生体情報PC2の相関を解析--------")
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, new_score2)
    # 描画する
    my2Dplot(np.vstack((q_un1, new_un2)).T, np.vstack((q_non1, new_non2)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information load>0.5 pc2")
    """
    
    
    
    
    
    """
    # 主観評価のPC1と生体情報のPC1との相関係数が高い指標だけをつかった解析#############################
    # 主観評価PC1と生体情報全指標の相関係数を計算
    # 計算用のデータフレームを作る
    cor_df = pd.concat([pd.DataFrame(q_transformed[:, 0], columns=["PC1"]), stan_data_df], axis=1, sort=False)
    # 相関データフレーム
    cor_df = cor_df.corr()["PC1"].drop("PC1")
    
    # 相関の絶対値が0.45以上の指標だけをつかって主成分分析
    # 相関の絶対値が0.45以上の指標を抜き出す
    new_stan_data_df = stan_data_df.drop(cor_df[np.fabs(cor_df)<0.45].index, axis=1)
    print("主観評価PC1との相関係数が0.45以上の生体情報をPCA-----------------------")
    new_transformed, new_un_df, new_non_df, new_loading = mypca1(new_stan_data_df.values, odor)
    print("主観評価PC1との相関係数が0.45以上の生体情報PC1")
    myloading_top3(new_loading[0], new_stan_data_df.columns) # pc1の因子負荷量
    print("主観評価PC1との相関係数が0.45以上の生体情報PC2")
    myloading_top3(new_loading[1], new_stan_data_df.columns) # pc2の因子負荷量
    
    ############ データ変数の準備 ############
    # 主観評価PC1との相関係数が0.45以上の生体情報PC1
    new_score1 = np.hstack((new_un_df.iloc[:, 0].values, new_non_df.iloc[:, 0].values))
    new_un1 = new_un_df.iloc[:, 0].values
    new_non1 = new_non_df.iloc[:, 0].values
    # 主観評価PC1との相関係数が0.45以上の生体情報PC2
    new_score2 = np.hstack((new_un_df.iloc[:, 1].values, new_non_df.iloc[:, 1].values))
    new_un2 = new_un_df.iloc[:, 1].values
    new_non2 = new_non_df.iloc[:, 1].values
    ########################################
    
    
    print("主観評価PC1との相関係数が0.45以上の生体情報PC1と生体情報PC2の相関を解析---")
    # 相関係数，p値を計算する
    correlation_analysis(new_score1, new_score2)
    # 描画する
    my2Dplot(np.vstack((new_un1, new_un2)).T, np.vstack((new_non1, new_non2)).T,
             "PCA scatter", "Biometric information cor>0.45 pc1", "Biometric information cor>0.45 pc2")

    print("主観評価PC1と主観評価PC1との相関係数が0.45以上の生体情報PC1の相関を解析---")
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, new_score1)
    # 描画する
    my2Dplot(np.vstack((q_un1, new_un1)).T, np.vstack((q_non1, new_non1)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information cor>0.45 pc1")

    print("主観評価PC1と主観評価PC1との相関係数が0.45以上の生体情報PC2の相関を解析---")
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, new_score2)
    # 描画する
    my2Dplot(np.vstack((q_un1, new_un2)).T, np.vstack((q_non1, new_non2)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information cor>0.45 pc2")









    # 主観評価のPC1と生体情報のPC2との相関係数が高い指標だけをつかった解析#############################
    # 主観評価PC1と生体情報全指標の相関係数を計算
    # 計算用のデータフレームを作る
    cor_df = pd.concat([pd.DataFrame(q_transformed[:, 1], columns=["PC2"]), stan_data_df], axis=1, sort=False)
    # 相関データフレーム
    cor_df = cor_df.corr()["PC2"].drop("PC2")
    
    # 相関の絶対値が0.2以上の指標だけをつかって主成分分析
    # 相関の絶対値が0.2以上の指標を抜き出す
    new_stan_data_df = stan_data_df.drop(cor_df[np.fabs(cor_df)<0.2].index, axis=1)
    print("主観評価PC1との相関係数が0.2以上の生体情報をPCA-----------------------")
    new_transformed, new_un_df, new_non_df, new_loading = mypca1(new_stan_data_df.values, odor)
    print("主観評価PC1との相関係数が0.2以上の生体情報PC1")
    myloading_top3(new_loading[0], new_stan_data_df.columns) # pc1の因子負荷量
    print("主観評価PC1との相関係数が0.2以上の生体情報PC2")
    myloading_top3(new_loading[1], new_stan_data_df.columns) # pc2の因子負荷量
    
    ############ データ変数の準備 ############
    # 主観評価PC1との相関係数が0.2以上の生体情報PC1
    new_score1 = np.hstack((new_un_df.iloc[:, 0].values, new_non_df.iloc[:, 0].values))
    new_un1 = new_un_df.iloc[:, 0].values
    new_non1 = new_non_df.iloc[:, 0].values
    # 主観評価PC1との相関係数が0.2以上の生体情報PC2
    new_score2 = np.hstack((new_un_df.iloc[:, 1].values, new_non_df.iloc[:, 1].values))
    new_un2 = new_un_df.iloc[:, 1].values
    new_non2 = new_non_df.iloc[:, 1].values
    ########################################
    
    
    print("主観評価PC1との相関係数が0.2以上の生体情報PC1と生体情報PC2の相関を解析---")
    # 相関係数，p値を計算する
    correlation_analysis(new_score1, new_score2)
    # 描画する
    my2Dplot(np.vstack((new_un1, new_un2)).T, np.vstack((new_non1, new_non2)).T,
             "PCA scatter", "Biometric information cor>0.2 pc1", "Biometric information cor>0.2 pc2")

    print("主観評価PC1と主観評価PC1との相関係数が0.2以上の生体情報PC1の相関を解析---")
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, new_score1)
    # 描画する
    my2Dplot(np.vstack((q_un1, new_un1)).T, np.vstack((q_non1, new_non1)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information cor>0.2 pc1")

    print("主観評価PC1と主観評価PC1との相関係数が0.45以上の生体情報PC2の相関を解析---")
    # 相関係数，p値を計算する
    correlation_analysis(q_score1, new_score2)
    # 描画する
    my2Dplot(np.vstack((q_un1, new_un2)).T, np.vstack((q_non1, new_non2)).T,
             "PCA scatter", "Subjective evaluation pc1", "Biometric information cor>0.2 pc2")
    """

 
# 【関数定義】 ##################################################################
# データフレーム用のインデックス名，カラム名を設定してくれる関数
def df_index_columns_name(df, i_name, c_name):
    index_name = []
    for i in range(1, len(df.index)+1):
        index_name.append(i_name + str(i)) # インデックス名リストを作成
    columns_name = []
    for i in range(1, len(df.columns)+1):
        columns_name.append(c_name + str(i)) # カラム名リストを作成
    df.index = index_name
    df.columns = columns_name
    return df

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
    data = sp.stats.zscore(df, axis=0)
    # データフレーム型に戻す
    df = pd.DataFrame(data)
    df.columns = columns # カラム名も戻す
    
    df = df.assign(No=task.values.tolist()) # タスク番号を標準化前のもので上書き
    df = df.assign(Stimulation=stimulation.values.tolist()) # 刺激の種類を付け足す
    
    return df

# 指標毎に標準化してあるデータを手に入れる関数 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def find_mydata():
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
    stan_data = sp.stats.zscore(data2_df, axis=0)
    # データフレーム型に変換
    stan_data_df = pd.DataFrame(stan_data, columns=COLUMNS48)
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
    q_stan_data = sp.stats.zscore(q_data2_df, axis=0)
    q_stan_data_df = pd.DataFrame(q_stan_data, columns=q_data_df.drop(["No", "Stimulation"], axis=1).columns)
    
    # -------------------------------------------------------------------------
    
    # 刺激の種類のndarray配列を用意
    odor = q_data_df["Stimulation"].values.tolist()
    odor = np.reshape(odor, (len(odor),1)) # 刺激の種類
    
    # 主観評価データ，生体情報データ，刺激の種類を返す
    return q_stan_data_df, stan_data_df, odor

# 与えられたデータを主成分分析して，匂い刺激別に分ける関数 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\    
def mypca1(data, odor):
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
    un_df = df_index_columns_name(pd.DataFrame(un), "Stim", "PC")
    non_df = df_index_columns_name(pd.DataFrame(non), "Stim", "PC")
    
    # 主成分の次元ごとの寄与率を出力する
    print("寄与率:\n", pca.explained_variance_ratio_, "\n")
    
    # 因子負荷量を出力する
    #print("PC1の因子負荷量：\n", loading[0], "\n")
    #print("PC2の因子負荷量：\n", loading[1], "\n")
    
    return transformed, un_df, non_df, loading

# 因子負荷量top3を抜き出す関数
def myloading_top3(loading, columns_name):
    loading_df = pd.Series(loading, index=columns_name)
    print("正の負荷量top3:\n", loading_df.sort_values(ascending=False)[0:3], "\n")
    print("不の負荷量top3:\n", loading_df.sort_values(ascending=True)[0:3], "\n")
    return

# 2つのデータの相関係数をprintする関数 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\    
def correlation_analysis(data1, data2):
    # 相関をscipyで計算
    cor = sp.stats.pearsonr(data1, data2)
    print("相関係数：\n", cor[0])
    print("p値：\n", cor[1], "\n")

# 2次元プロットする関数(データ1，データ2，グラフタイトル，x軸ラベル，y軸ラベル) \\\\\\\\\\\\\\\\\
def my2Dplot(data1, data2, tytle, x, y):
    
    # グラフ描画サイズを設定する
    plt.figure(figsize=(5, 5))

    # 主成分をプロットする
    plt.scatter(data1[:, 0], data1[:, 1], s=80, c=[0.4, 0.6, 0.9], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.scatter(data2[:, 0], data2[:, 1], s=80, c=[0.5, 0.5, 0.5], alpha=0.8, linewidths="1", edgecolors=[0, 0, 0])
    plt.title(tytle, fontsize=18)
    plt.xlabel(x, fontsize=18)
    plt.ylabel(y, fontsize=18)
    
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
