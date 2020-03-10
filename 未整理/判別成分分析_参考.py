# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:21:22 2018

@author: A_lifePC
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import precision_recall_fscore_support  as prf


digits = load_digits()

gnb = GNB()

print(gnb)

df = pd.DataFrame([], columns=[
    "n_components",
    "pca-gnn precision", "pca-gnn recall", "pca-gnn f1",
    "lda-gnn precision", "lda-gnn recall", "lda-gnn f1"])
for n_components in [5, 10, 15, 20, 25, 30, 40]:
    pca = PCA(n_components=n_components) # 主成分分析
    lda = LDA(n_components=n_components) # 線形判別分析

    # zipは複数のリストの要素をまとめて取得
    steps1 = list(zip(["pca", "gnb"], [pca, gnb]))
    steps2 = list(zip(["lda", "gnb"], [lda, gnb]))

    p1 = Pipeline(steps1)
    p2 = Pipeline(steps2)

    score_lst = []
    for decomp_name, clf in zip(["pca", "lda"], [p1, p2]):
        trues = []
        preds = []
        for train_index, test_index in SKF(
                shuffle=True, random_state=0).split(
                digits.data, digits.target):
            clf.fit(digits.data[train_index], 
                    digits.target[train_index])
            trues.append(digits.target[test_index])
            preds.append(clf.predict(digits.data[test_index]))
        scores = prf(np.hstack(trues), np.hstack(preds), average="macro")
        score_lst.extend(scores[:-1])
    df = df.append(pd.Series([n_components, *score_lst],
                             index=df.columns),
                   ignore_index=True)
print(df)
df.plot(x="n_components", y=["pca-gnn f1", "lda-gnn f1"])
plt.savefig("判別成分分析_参考.png")

