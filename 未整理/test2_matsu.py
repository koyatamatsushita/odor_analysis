# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:59:58 2018

@author: A_lifePC

中心差分法の微分のテスト
cosを微分しています(sinを出力できるのが理想)
"""

import math
import matplotlib.pyplot as plt
import numpy as np

def f(t):
    return math.cos(t)

DELTA_T = 0.001
MAX_T = 100.0

t = 0.0 # t初期値
x = 0.0 # t=0でのx

x_hist = [x]
t_hist = [t]

while t < MAX_T:
    x += 2*f(t-DELTA_T)*DELTA_T
    t += 2*DELTA_T
    x_hist.append(x)
    t_hist.append(t)

# 数値解のプロット 
plt.plot(t_hist, x_hist)

# 厳密解(sin(t))のプロット
t = np.linspace(0, MAX_T, 1/DELTA_T)
x = np.sin(t)
plt.plot(t, x)

plt.xlim(0, MAX_T)
plt.ylim(-1.3, 1.3)

plt.show()