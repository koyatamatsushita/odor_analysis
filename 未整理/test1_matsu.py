# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:12:31 2018

@author: A_lifePC
"""

# とりあえずExcelファイルに生体信号を記録したものを読み込んで，
# β，ηの各指標(全8項目)について散布図を作成したい．
# 読み込むデータは
# ・主観評価の主成分スコア
# ・β，ηの各標準化済みデータ
# の2つで，これを三次元の散布図にマッピングする．

#3Dmap参考：https://qiita.com/mojaie/items/c993fbb3aa63d0001c1c


# 【import】 ####################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 【前処理】 ####################################################################
# データがあるパスとそのファイル名
DATAPATH = "C:\\Users\\A_lifePC\\Desktop\\ファイル読み込み用"
FILENAME = "2018_11_data.csv"
# 不快臭のデータ数
UN_NUM = 25


# 【メイン処理】 ##################################################################
"""main関数"""
def main():
    #　データがあるパスに作業ディレクトリ変更
    os.chdir(DATAPATH)
    # csvの読み取り
    data = pd.read_csv(FILENAME, header=0)
    
    # 三次元プロット（主観評価，β，η）
    #scat3d(data)
    # グラフ作成
    fig = plt.figure()
    ax = Axes3D(fig)
    
    # 軸ラベルの設定
    ax.set_xlabel("eta")
    ax.set_ylabel("beta")
    ax.set_zlabel("emotion")
    
    # 表示範囲の設定
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    
    # 不快臭(d1)，無臭(d2)のデータで色分け
    d1 = data.iloc[0:UN_NUM,:]
    d2 = data.iloc[UN_NUM:,:]
    
    # グラフ描画
    # ms:点の大きさ mew:点の枠線の太さ mec:枠線の色
    # 4列目はetamax, 0列目はbetamax, 8列目はemotion
    ax.plot(d1.iloc[:,4], d1.iloc[:,0], d1.iloc[:,8],
            "o", color=(0.2,0.7,1), ms=6, mew=1, mec='0.0')
    ax.plot(d2.iloc[:,4], d2.iloc[:,0], d2.iloc[:,8], 
            "o", color=(0.7,0.7,0.7), ms=6, mew=1, mec='0.0')
    plt.show()
    
    """
    # 自動回転
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
   """
    
# 【関数定義】 ##################################################################
"""グラフを作る関数
def scat3d(data):
    # グラフ作成
    fig = pyplot.figure()
    ax = Axes3D(fig)
    
    # 軸ラベルの設定
    ax.set_xlabel("emotion")
    ax.set_ylabel("beta")
    ax.set_zlabel("eta")
    
    # 表示範囲の設定
    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    ax.set_zlim(-4,4)
    
    # 不快臭(d1)，無臭(d2)のデータで色分け
    d1 = data[0:UN_NUM-1,:]
    d2 = data[UN_NUM:,:]
    
    # グラフ描画
    ax.plot(d1[:,0], d1[:,4], d1[:,8], "o", color=(0,0,1), ms=4, mew=0.5)
    ax.plot(d2[:,0], d2[:,4], d2[:,8], "o", color=(0.3,0.3,0.3), ms=4, mew=0.5)
    pyplot.show()
"""
    
# 【main実行】 ##################################################################
if __name__ == '__main__':
    main()
    