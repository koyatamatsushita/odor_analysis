# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:00:06 2018

@author: A_lifePC
"""
import numpy as np
import pandas as pd

data = np.zeros([51, 8])
data_df = pd.DataFrame(data)

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

index_name, columns_name = df_index_columns_name(data_df, "Stim", "PC")