import numpy as np
class HL:
    """
    目的:
        波形データから，ピーク値評価後，レインフロー計数を行う
        アルゴリズムはヒステリシスループ法による
    例題:
        hl=HL()
        wave=hl.demo_data()
        hl.SetWave(wave)
        hl.Calc()
        halfR,halfM=hl.GetRes()
        print('half range=',halfR)
        print('half mean=',halfM)
    """
    def __init__(self):
        self.Wave=[]
        self.num=0
        self.Peak=[]
        self.halfR=[]
        self.halfM=[]
    def SetPeak(self,peak):
        """
        目的:リストで与えられるピーク値列をself.Peakにセットする
        """
        self.Peak=peak
    def demo_data(self):
        """
        目的:
            デモ用のサンプル波形データのリストを戻す
        """
        return [5e-3,3.2e-2,3.8e-2,-3.3e-2,-1.9e-2,-1e-2,1e-3,-8e-3,-2e-2,1e-2,-1e-3,4e-3,1.1e-2,-1e-3,-7e-3,-2e-3]
    def SetWave(self,wave):
        """
        目的:
            波形データをインスタンスにセット
        入力:
            wave  波形データのリスト
        """
        self.Wave=wave
        self.num=len(wave)
    def Calc(self):
        """
        目的:
           登録されている波形データに対し，ピーク値計算，波形計数処理を行う
        """
        self.PeakCalc()
        self.halfR,self.halfM=self.hloop()
    def GetRes(self):
        """
        目的:
            波形計数処理後のデータの取得
        出力:
            halfR    半波のリスト
            halfM    半波の平均値
        """
        return self.halfR,self.halfM
    def GetPeak(self):
        """
        目的:
            計算されたピーク値リストの取得
        """
        return self.Peak
    def peak(self,y0,y1,y2):
        """
        目的:
            相続く3点の波形データからビーク値判定を行い，ピーク値を戻す
        入力:
            y0,y1,y2    相続く3点の波形データ
        出力:flag,time,value
            flag        ピーク値が判定されるときTrue,それ以外False
            time        ピークの判定される時間
            value       放物線近似により評価されたピーク値
        """
        pflag = (y2-y1)*(y1-y0)
        flag=False
        time=0.0; value=0.0
        if pflag<=0:
            a= (y0+y2-2*y1)/2.0
            b= -(y0-y2)/2.0
            c= y1
            time= -b/2/a
            value= c-b*b/4.0/a
            flag=True
        return flag,time,value
    def PeakCalc(self):
        """
        目的:
            self.Waveのピーク値評価を行い結果をself.Peakに入力
        """
        n=self.num
        m=n
        wn=self.Wave
        j=0
        wn[j]=self.Wave[0]
        for i in range(1,n):
            if self.Wave[i] != wn[j]:
                j += 1
                wn[j] = self.Wave[i]
        num=j
        for i in range(num-1):
            flag,time,value=self.peak(wn[i],wn[i+1],wn[i+2])
            if flag==True:
                self.Peak.append(value)
    def hloop(self):
        """
        目的:
            self.Peakに対してヒステリシスループ法を適用し，半波の情報を戻す
        出力:res_r,res_m
            res_r    半波のレンジのリスト
            res_m    半波の平均値のリスト
        """
        peak_num=len(self.Peak)
        p=np.zeros(peak_num)
        res_r=[]
        res_m=[]
        j=0
        for i in range(peak_num):
            j += 1
            pk1=self.Peak[i]
            p[j]=pk1
            while j>2:
                r1=np.abs(p[j-1]-p[j-2])
                r2=np.abs(p[j]-p[j-1])
                if r1>r2:
                    break
                r=r1
                m=(p[j-1]+p[j-2])/2
                res_r.append(r)
                res_r.append(r)
                res_m.append(m)
                res_m.append(m)
                j=j-2
                p[j]=pk1
        for i in range(1,j):
            r=np.abs(p[i+1]-p[i])
            m=(p[i+1]+p[i])/2.0
            res_r.append(r)
            res_m.append(m)
        return res_r,res_m        