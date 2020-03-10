# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:49:49 2018

@author: A_lifePC
"""

from numpy import *
from matplotlib.pyplot import *
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import DictionaryLearning

# 混合ガウス分布に従う乱数の生成
def rand_gauss_mix(mu, sigma, p, N):
	d, K = mu.shape
	p_cumsum = cumsum(array(p) / sum(p))
	X = zeros([d, 0])
	for n in range(N):
		p_ = p_cumsum - random.rand()
		p_[p_<0] = 1
		k = p_.argmin()
		x = random.multivariate_normal(mu[:,k], sigma[:,:,k]).reshape(-1,1)
		X = hstack((X, x))
	return X

# 回転行列
Rot = lambda rad: array([[cos(rad), sin(rad)], [-sin(rad), cos(rad)]])


# データの生成
N = 1000
distribution_type = 'rectangle'

if distribution_type == 'gauss':
	mu = array([0, 0])
	sigma = array([[0.1,0.1],[0.2,0.3]])
	X = random.multivariate_normal(mu, sigma, N).T
elif distribution_type == 'rectangle':
	rad = (1./6)*pi
	ext = [1, 0.4]
	X = 2*random.rand(2,N)-ones([2,N])
	X = dot(Rot(rad), X)
	X = dot(diag(ext), X)
elif distribution_type == 'closs':
	mu = zeros([2,2])
	p = [0.5, 0.5]
	sigma0 = diag([0.3,0.003])
	rad = [1./8*pi, 7./8*pi]
	sigma = zeros([2, 2, 2])
	for i in range(2):
		sigma[:,:,i] = dot( Rot(rad[i]), dot(sigma0, Rot(rad[i]).T) )
	X = rand_gauss_mix(mu, sigma, p, N)

# 主成分分析
decomposer = PCA()
decomposer.fit(X.T)
Upca = decomposer.components_.T
Apca = decomposer.transform(X.T).T

# 独立成分分析
decomposer = FastICA()
decomposer.fit(X.T)
Uica = decomposer.mixing_ 
Aica = decomposer.transform(X.T).T

# スパースコーディング
decomposer = DictionaryLearning()
decomposer.fit(X.T)
Usc = decomposer.components_ .T
Asc = decomposer.transform(X.T).T


axis('equal')
plot(X[0],X[1],'xc')
Upca = Upca / sqrt((Upca**2).sum(axis=0))
Uica = Uica / sqrt((Uica**2).sum(axis=0))
Usc = Usc / sqrt((Usc**2).sum(axis=0))
for i in range(2):
    p_pca = plot([0, Upca[0,i]], [0, Upca[1,i]], '-r')
    p_ica = plot([0, Uica[0,i]], [0, Uica[1,i]], '-b')
    p_sc = plot([0, Usc[0,i]], [0, Usc[1,i]], '-g')
legend(('data', 'PCA', 'ICA', 'SC'))
legend(loc="best", prop=dict(size=12))
show()

subplot(1,3,1)
plot(Apca[0], Apca[1], 'xc')
title('PCA')
subplot(1,3,2)
plot(Aica[0], Aica[1], 'xc')
title('ICA')
subplot(1,3,3)
plot(Asc[0], Asc[1], 'xc')
title('SC')
show()