# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:20:32 2018

@author: A_lifePC

線形判別分析勉強用．
参考：(http://robonchu.hatenablog.com/entry/2017/10/16/220705)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 使用するデータをダウンロード
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

print(df_wine.head()) # head()で先頭5行分を確認できる．

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 機械学習用の処理．トレーニングデータとテストデータに分割する．
# train_test_splitで，ランダムに好きな割合で分割可能．
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3,
                     random_state=0) # random_state：データ分割用の乱数のシード値

from sklearn.preprocessing import StandardScaler

# StandardScalerで標準化
# fit_transformにデータを渡すと標準化してくれます．
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)



# クラスごとに，平均ベクトルを計算

# 表示のフォーマットを指定する（precisionで有効桁を指定）
np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    # yが1~3の条件でxを抜き出し，各列ごとに平均を計算する
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))

# クラス間変動行列Sbと，クラス内変動行列Swを生成

# クラス内変動行列    

# !!!冗長な処理
d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))  # scatter matrix for each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
        # dotは内積計算． np.dot((row - mv), (row - mv).T)でもおなじこと
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter                          # sum class scatter matrices

print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
# Within-class scatter matrix: 13x13 と表示されるはず
# !上記の処理は，クラスラベルが一様に分布していないといけない

# クラスラベルの確認
print('Class label distribution: %s'    
   % np.bincount(y_train)[1:]) # bincountは負でない整数配列の各値の数をカウントする
# Class label distribution: [40 49 35]　と表示されるはず
# !クラスラベル数がそれぞれ異なるので，スケーリングが必要．
# スケーリングしてサンプル数で割る操作を行うと，共分散行列と一致

# 上記処理と同じ結果となる（共分散行列）
d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0],
                                                     S_W.shape[1]))

# クラス間変動行列
mean_overall = np.mean(X_train_std, axis=0)
d = 13  # number of features
S_B = np.zeros((d, d))
# enumerateで要素とインデックス番号を同時に取得できる
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0] # shapeで要素数を取得
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    mean_overall = mean_overall.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))
# Between-class scatter matrix: 13x13　と出力されるはず


# 行列Sw^-1 * sbの固有ベクトルと対応する固有値を計算する
# 固有値問題を解く
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# Make a list of (eigenvalue, eigenvector) tuples
# for文のリスト内包表記
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
# lambda は無名関数の定義
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])


# 線形判別のプロット
    
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/lda1.png', dpi=300)
plt.show()


# d x k次元の変換行列Wを生成するために最も大きいk個の固有値に対応する
# k個の固有ベクトルを選択する（固有ベクトルは行列の列）
# 変換行列の作成
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('\nMatrix W:\n', w)


# トレーニングデータセットの変換
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

plt.figure()

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0] * (-1),
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
# plt.savefig('./figures/lda2.png', dpi=300)
plt.show()


# 以下，scikit-learnによってLDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda2 = lda.fit_transform(X_train_std, y_train)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train_lda2, y_train)

