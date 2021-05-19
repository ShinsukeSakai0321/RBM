import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
from sklearn.preprocessing import StandardScaler
from RBM import reliability as rel
class SensitivityAnal:
    """
    DataFrameで与えられるX,yをもとに、酒井理論でSensitivityを計算する
    下記例では、項目名、感度　を感度の昇順でDataFrame resultに与えられる
    使い方
        sa=SensitivityAnal(X,y)
        result=sa.GetResult(pa)
    """
    def __init__(self,X,y,gamma=0.001,C=10):
        scaler = StandardScaler()
        scaler.fit(X)
        data_std=scaler.transform(X)
        v_mean=scaler.mean_
        v_std=np.sqrt(scaler.var_)
        train_x=data_std
        var_num=train_x.shape[1]
        train_t1=y
        # SVMによる解析        
        from sklearn.svm import SVC
        svm=SVC(kernel='rbf',C=C,gamma=gamma,probability=True).fit(train_x,train_t1)
        sv=svm.support_vectors_
        dp,beta=rel.SV_RF(sv,svm,gamma)
        self.beta=beta
        self.Dp=dp*v_std+v_mean
        self.N_s=sum(train_t1)
        self.N_f=len(train_t1)-self.N_s
        alpha,deriv=rel.sv_alpha(svm,gamma,dp)
        factor_ind=np.argsort(alpha)
        name=X.columns
        res=[]
        for i in range(len(name)):
            res.append([name[factor_ind[len(name)-i-1]],alpha[factor_ind[len(name)-i-1]]])
        res=pd.DataFrame(res)
        res.columns=['categ','sensitivity']
        self.result=res
    def GetResult(self,pa):
        """
        項目についてを感度の降順に示す。
        入力
            pa　PreAnalのインスタンス
        """
        res=self.result
        table=pa.GetCategTable()
        cont=[]
        for i in range(len(res)):
            try:
                aa=table[table['index']==res.iloc[i,0]]['contents']
                cont.append(aa.iloc[0])
            except:
                cont.append('NaN')
        res['answer']=cont
        return res
    def GetBeta(self):
        return self.beta
    def Next(self,thres,res):
        """
        dataframe resの第一列目与えられる項目のうち、第二列目に与えられる感度がthres以上の項目を抽出し、リストとして返す
        """
        next_features=[]
        for i in range(len(res)):
            if abs(res.iloc[i,1])>thres:
                next_features.append(res.iloc[i,0])
        return next_features
    def NextTop(self,top,res):
        """
        dataframe resの第一列目与えられる項目のうち、第二列目に与えられる感度絶対値がtop番目までの項目を抽出し、リストとして返す
        """
        next_features=[]
        aa=res.copy()
        bb=abs(aa['sensitivity'])
        aa['sensitivity']=bb
        cc=aa.sort_values(by='sensitivity', ascending=False)
        for i in range(top):
            next_features.append(cc.iloc[i,0])
        return next_features     
    def PickTerm(self,Sout):
        """
        感度分析結果について，Soutのリストに与える項目の結果を
        集約してDataFrameとして戻す
        """
        tt = pd.DataFrame(index=[])
        for term in zip(Sout):
            tt=tt.append(self.result[(self.result['categ']==term[0])])
        return tt
    def ModRes(self,result,contents):
        """
        感度分析結果resultに対して、項目名をcontentsから抽出し、resultに対して項目名'term'として追加して返す
        """
        term=[]
        resu=result.copy()
        for i in range(len(result)):
            aa=result['categ'][i][:2]
            term.append(contents[(contents['C-1']==aa)].iloc[0,1])
        resu['term']=term
        return resu