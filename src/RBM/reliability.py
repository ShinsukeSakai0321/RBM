"""
目的:SVMを利用した信頼性評価モジュール
注意点:モジュール利用前に，SVM解析を終了していること
"""
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
def UnderSampling4(XX,DP,rpick,ratio):
    #DP周りの半径rpick内について、ratioの率だけ削減してサンプル
    d=np.zeros(len(XX))
    for i in range(len(XX)):
        d[i]=eval_dist(XX.iloc[i],DP)
    XX['d']=d
    XX2=XX[XX['d']<rpick]
    XX.drop(columns=['d'],inplace=True)
    XX2.drop(columns=['d'],inplace=True)
    return XX2
#サポートベクターマシンα，偏微分ベクトルの計算
import math
def sv_alpha(svm,gamma,x0):
    """
    目的:座標点x0の感度係数と，微分値を計算する
    入力:
        svm SVM解析の出力
        gamma   SVM解析の際に用いたγ値
        x0      評価点の座標
    出力:alpha,deriv
        alpha   感度ベクトル
        deriv   微分値ベクトル
    """
    #derivative
    var_num=len(x0)
    coef=svm.dual_coef_
    sv=svm.support_vectors_
    b=svm.intercept_
    # value of kernel function
    nsv=len(sv)
    kernel=np.zeros(nsv)
    #x0=np.array(dp)
    for i in range(nsv):
        ee=-gamma*np.linalg.norm(x0-sv[i])**2
        kernel[i]=math.exp(ee)
    #derivative
    deriv=np.zeros(var_num)
    for j in range(var_num):
        ar_d=(-2.*gamma)*coef*(x0[j]-sv[:,j])
        deriv[j]=np.dot(ar_d,kernel.T)
    alpha=deriv/np.linalg.norm(deriv)
    # alpha: 感度係数
    # deriv: 各変数方向の微分値
    return alpha,deriv
def g(x,svm):    # svmでのgのsurrogate関数
    """
    目的:SVMの解析結果から，x点のdecision_functionの値を返す
    入力:
        x   評価点
        svm SVM解析の出力
    出力:
        decisio_functionの値
    """
    var_num=len(x)
    return svm.decision_function(x.reshape(1,var_num))[0]
def RackwitzFiessler(x,svm,gamma,nmax=100,eps=0.001,b0=10.0):
    """
    目的:初期点xからスタートし，Rackwitz Fiessler法により設計点を求める
    入力:
        x       初期点
        svm     SVM解析の出力
        gamma   SVM解析の際に用いたγ値
        nmax    繰り返し数上限
        eps     収束規準
        b0      beta値初期値
    出力:dp,beta
        dp      設計点
        beta    信頼性指標
    """
    for i in range(nmax):
        alpha,nabla=sv_alpha(svm,gamma,x)
        if(np.isnan(alpha[0])):
            beta=1e4
            break
        x2=1/np.dot(nabla,nabla)*(np.dot(nabla,x)-g(x,svm))*nabla.T
        x=x2
        beta=np.linalg.norm(x)
        de=abs((beta-b0)/b0)
        b0=beta
        if de<eps:
            break
    dp=x
    return dp,beta
def SV_RF(sv,svm,gamma):
    """
    目的:全SV点を出発点とするRFを実施し、最小βを与える点を設計点と確定
    入力:
        sv  SVM解析の結果得られるsv点
        svm     SVM解析の出力
        gamma   SVM解析の際に用いたγ値
    出力:dp,beta
        dp      原点から最短の設計点
        beta　  β値
    """
    beta=np.zeros(len(sv))
    for i in range(len(sv)):
        dp,beta[i]=RackwitzFiessler(sv[i],svm,gamma)
    start=sv[np.argmin(beta)]
    return RackwitzFiessler(start,svm,gamma) 
from pyDOE import *
from scipy.stats.distributions import norm
def generate_datablock3(n,Mr,Sr,Ms,Ss,k):
    """
    目的:RSモデルについて，LHS+USによるサンプル点の発生
    入力:
        n   発生点数
        Mr  強度平均値
        Sr  強度標準偏差
        Ms  荷重平均値
        Ss  荷重標準偏差
        k   USの領域設定のための係数
    出力:
        df  列名['x1','x2','t']のデータフレーム
    """
    #Latin Hypercube Sampling
    design = lhs(2, samples=n,criterion='maximin')
    u=design
    means = [Mr,Ms]
    stdvs = [Sr,Ss]
    for i in range(2):
        u[:, i] = k*stdvs[i]*(2*design[:,i]-1)+means[i]
    dR=u[:,0]
    dS=u[:,1]
    df = DataFrame(dR, columns=['x1'])
    df['x2']=dS
    df['t']=0
    df['t']=(np.sign(dR-dS)+1)/2
    return df
