# -*- coding: utf-8 -*-
"""
C++の結果から，Arduino信号を検出して不要なデータを消し，
HPF
LPF
標準化などを行う

【注意！！】
    プログラム実行前に，MakeNewData.mを使用しましょう


[参考文献]
人工知能に関する断創録
http://aidiary.hatenablog.com/entry/20120103/1325594723

"""
# デバッグ用
# import pdb; 
# pdb.set_trace()

# 【import】 ####################################################################
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import math
import matplotlib.pyplot as plt                



# 以下関数 ##################################################################### 

# fft
def fft(f,s,r):#f:データ、s:サンプリング周期、r:時間データ
    N_data = len(f)
    #fs=1000/s
    fs=1/s   
    N = 2**11    # FFTのサンプル数
    # 最低、N点ないとFFTできないので0.0を追加
    for i in range(N-N_data): f.append(0.0)
    #print('2のべき乗の長さに変更後',len(f))    
    Window1 = np.hamming(N)
#    Window2 = np.hanning(N)   
    f_w = Window1*f
        # 高速フーリエ変換
    F = np.fft.fft(f_w)/(N/2)
        # 直流成分の振幅を揃える
    #F[0] = F[0]/2
        # 振幅スペクトル
    amp = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in F]  
        # 位相スペクトル
    phase = [np.arctan2(int(c.imag), int(c.real)) for c in F]    
        # 周波数軸の値を計算 
    freq = np.fft.fftfreq(N,d=1.0/fs)
        # ナイキスト周波数の範囲内のデータのみ取り出し
    freq = freq[1:int(N/2)]
    amp = amp[1:int(N/2)]
    phase = phase[1:int(N/2)]
    
    return freq,amp


def createLPF(fc):
    """IIR版ローパスフィルタ、fc:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2
    b[0] = (4 * np.pi**2 * fc**2) / denom
    b[1] = (8 * np.pi**2 * fc**2) / denom
    b[2] = (4 * np.pi**2 * fc**2) / denom
    a[0] = 1.0 #コードでは使わないけど，1.0を代入
    a[1] = (8 * np.pi**2 * fc**2 - 2) / denom
    a[2] = (1 - (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2) / denom
    return a, b

def createHPF(fc):
    """IIR版ハイパスフィルタ、fc:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2
    b[0] = 1.0 / denom
    b[1] = -2.0 / denom
    b[2] = 1.0 / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc**2 - 2) / denom
    a[2] = (1 - (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2) / denom
    return a, b

def createBPF(fc1, fc2):
    """IIR版バンドパスフィルタ、fc1、fc2:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2
    b[0] = (2 * np.pi * (fc2 - fc1)) / denom
    b[1] = 0.0
    b[2] = - 2 * np.pi * (fc2 - fc1) / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc1 * fc2 - 2) / denom
    a[2] = (1.0 - 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2) / denom
    return a, b

def iir(x, a, b):
    """IIRフィルタをかける、x:入力信号、a, b:フィルタ係数"""
    y = [0.0] * len(x)  # フィルタの出力信号
    Q = len(a) - 1
    P = len(b) - 1
    for n in range(len(x)):
        for i in range(0, P + 1):
            if n - i >= 0:
                y[n] += b[i] * x[n - i]
        for j in range(1, Q + 1):
            if n - j >= 0:
                y[n] -= a[j] * y[n - j]
    return y


# 3次スプライン補完でリサンプリング
def resmp(time,data):
#    time = df_out['newTime']
#    data = df_out['β']
    
    f_CS = interp1d(time, data, kind='cubic')  # 3次スプライン補完
    timerange = range(math.ceil(time[0]),math.floor(max(time)))    #　時間軸を整数化
    # math.ceilは切り上げ、math.floorは切り捨て    

    time_resmp = pd.Series([i for i in timerange])
    data_resmp = pd.DataFrame(([f_CS(i) for i in timerange]))   # 整数時間に沿ってリサンプリング    
    
    return time_resmp, data_resmp


def HPF(fs, fc, data):    
    fc_analog = np.tan(fc * np.pi / fs) / (2 * np.pi) 
    a, b = createHPF(fc_analog)
    # 順方向フィルタ
    data_f           = iir(data, a, b)          
    data_f_rev       = list(reversed(data_f))   # 反転
    # 逆方向フィルタ
    data_f_rev_f     = iir(data_f_rev, a, b)    
    data_f_rev_f_rev = list(reversed(data_f_rev_f))
    #　フィルタ後、反転させて元の順に戻す。これがゼロ位相フィルタ後。
    
    data_HPF = data_f_rev_f_rev
    
    return data_HPF

def LPF(fs, fc, data):    
    fc_analog = np.tan(fc * np.pi / fs) / (2 * np.pi) 
    a, b = createLPF(fc_analog)
    # 順方向フィルタ
    data_f           = iir(data, a, b)          
    data_f_rev       = list(reversed(data_f))   # 反転
    # 逆方向フィルタ
    data_f_rev_f     = iir(data_f_rev, a, b)    
    data_f_rev_f_rev = list(reversed(data_f_rev_f))
    #　フィルタ後、反転させて元の順に戻す。これがゼロ位相フィルタ後。
    
    data_LPF = data_f_rev_f_rev
    
    return data_LPF

# 標準化
def std(x):
    xmean = x.mean()
    xstd  = np.std(x)
    zscore = (x-xmean)/xstd    

    return zscore


# 【前処理(data_extract.pyと共通)】 ################################################

# データがあるパスに移動
Datapath = 'D:\\totsuka\\backup\\01.発表関連\\03.M2\\180919_COI研究会\\実験データ\\血管剛性まとめ'
# os.getcwd()

Subnames = os.listdir(Datapath)

# 出力フォルダ
Datapath2 = 'D:\\totsuka\\backup\\01.発表関連\\03.M2\\180919_COI研究会\\実験データ'
if not os.path.isdir(Datapath2+'\\'+'result_ToMakeGraph'):
    os.mkdir(Datapath2 + '\\' + 'result_ToMakeGraph')
Datapath_out = Datapath2 + '\\' + 'result_ToMakeGraph'


# 【メイン処理】 ################################################################
# 以下，Subnamesの分だけforする予定
# for i in range(0,10):
# for i in range(4,10):    
for i in range(len(Subnames)):        
    os.chdir(Datapath)

    #　DataFrameとしてベータの解析結果をよみとる
    databook = pd.ExcelFile(Subnames[i])
    
    df_rawdata = pd.DataFrame()
    df_rawdata = databook.parse(databook.sheet_names[0])#複数シートの場合
        
# -------------------------------------------------------------------------
# 解析に必要なデータだけ抽出　
# Arduino信号を見て，実験区間を抜き出す
# -------------------------------------------------------------------------    
    Ard = max(df_rawdata['イベント情報'])/6 # 閾値
    
    # Ard信号が閾値以上のときFLAGを立てる    
    df_rawdata['FLAG'] = ( ( df_rawdata['イベント情報'] > Ard ) ).astype(int) 
    ind = [] 
    ind = df_rawdata[df_rawdata['FLAG']==1].index

    # 連続してFLAGが立っている場合，先に立った方を真値（ind_new）とする        
    num = 0 # ind_newのカウンタ
    ind_new = np.zeros(25)    
    for j in range(len(ind)-1):        
        if not ind[j+1] - ind[j] == 1:
            ind_new[num] = ind[j]
            num = num + 1
            
    # Ardにノイズが入るとしたら，実験開始時のみなので，indの最終値は必ずrest2終了を表す
    ind_new[19] = ind[(len(ind)-1)]

# 例外処理 -----------------------------------------------
# nakagakiのデータだけ最後のFLAGが入っていなかったため例外処理
    if i == 5:
        ind_new[19] = df_rawdata.index.get_loc(len(df_rawdata)-1)
# -------------------------------------------------------        
    
    # うまく検出できれていれば，ind_newは20要素になる
    # (アンケート1開始,rest1開始,task開始×15,アンケート2開始,rest2開始，rest2終了)
    
    # ind_newが20要素以上のとき，実験開始信号をzureの分だけずらす
    if num > 19:
        print(Subnames[i] + 'is error')
        print(num)
        zure = num - 19
    else:    
        zure = 0
            
    # 備忘録
    # ind_newにはArd信号が入るべき場所のindexが入っている
        
#    # Ardデータの再定義            
#    df_rawdata['newArd'] = np.zeros(len(df_rawdata))
#    for k in range(0,20):
#        df_rawdata['newArd'][df_rawdata.index == ind_new[k]] = 5
                            
    # 不要なデータをカット    
    df_rawdata = df_rawdata[int(ind_new[zure]):int(ind_new[19])+1]        
    df_rawdata = df_rawdata.reset_index()
        
    # 出力データのまとめ
    newtime = np.zeros(len(df_rawdata))
    for k in range(len(df_rawdata)-1):
        # 新しい時間軸の設定
        newtime[k+1] = df_rawdata['Time'][k+1] - df_rawdata['Time'][k] + newtime[k]

    df_out = pd.DataFrame()
    df_out['newTime'] = newtime
    df_out['Time'] = df_rawdata['Time']    
    df_out['決定係数'] = df_rawdata['決定係数(PLS>0，提案モデル(誤差項抜き))']        
    df_out['β'] = df_rawdata['β']        
    df_out['イベント情報'] = df_rawdata['イベント情報']    

# -------------------------------------------------------------------------
# 各種解析
# -------------------------------------------------------------------------    

    #べーたの決定係数0.9未満を削除する．
    cd1 = df_out['決定係数']
    cd_remove_index = [s for (s,n) in enumerate(cd1) if n<0.9]
    # 決定係数0.9未満のインデックスのリストを作成。sはインデックス、nはそのインデックスの要素の値

    for ii in range(len(cd_remove_index)):
        df_out = df_out.drop(cd_remove_index[ii])
        #べーたの決定係数0.9未満のところを削除
        
    df_out = df_out.reset_index()

    # リサンプリング
    time_resmp, data_resmp = resmp(df_out['newTime'], df_out['β'])
    df_out_resmp = pd.DataFrame({'resmp_time': time_resmp}) # リサンプリング用のDataFrame         
    df_out_resmp['resmp_β'] = data_resmp      
    
    # フィルタ処理
    fs = 1 # resmpしてるからサンプリング周波数は1[Hz]
    fc = 0.15
    df_out_resmp['HPF_β'] = HPF(fs, fc, df_out_resmp['resmp_β'])
    df_out_resmp['LPF_β'] = LPF(fs, fc, df_out_resmp['resmp_β'])
            
    # 標準化
    df_out_resmp['resmp_β_std'] = std(df_out_resmp['resmp_β'])
    df_out_resmp['HPF_β_std'] = std(df_out_resmp['HPF_β'])
    df_out_resmp['LPF_β_std'] = std(df_out_resmp['LPF_β'])    

    # イベント情報の追加
    time_resmp, Ard_resmp = resmp(df_out['newTime'], df_out['イベント情報'])
    df_out_resmp['resmp_Ard'] = Ard_resmp    

# -------------------------------------------------------------------------
# excel出力
# -------------------------------------------------------------------------  
    
    if not os.path.isdir(Datapath_out + '\\' + 'RAWdata'):
        os.mkdir(Datapath_out + '\\' + 'RAWdata')
    os.chdir(Datapath_out + '\\' + 'RAWdata')      
    
    # ファイル名決定
    filePath1 = Subnames[i] + '_ToMakeGraph1.xlsx'    
    writer = pd.ExcelWriter(filePath1)    
    df_out.to_excel(writer,index = False)     
    writer.save()        

    if not os.path.isdir(Datapath_out + '\\' + 'HPFLPF'):
        os.mkdir(Datapath_out + '\\' + 'HPFLPF')
    os.chdir(Datapath_out + '\\' + 'HPFLPF')  
    # ファイル名決定    
    filePath2 = Subnames[i] + '_ToMakeGraph2.xlsx'    
    writer = pd.ExcelWriter(filePath2)    
    df_out_resmp.to_excel(writer,index = False)     
    writer.save()        
    
    print(Subnames[i] + '完了！')
    
# 画像としてグラフ作成
#    graph_df = df_out.loc[:,['newTime','β','イベント情報']]
#    graph_df.plot(x = ['newTime'], secondary_y = ['イベント情報'])
#    plt.title(Subnames[i])
#    plt.savefig(Subnames[i] + 'png')
    
    
print('完了！')






