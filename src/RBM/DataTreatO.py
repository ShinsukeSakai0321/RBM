""" DataTreat.pyのオブジェクト指向版
  UniPlannerデータをAI用に加工する
  GitLabでのソース管理用
"""
from os.path import dirname, exists, expanduser, isdir, join, splitext
import pandas as pd
import numpy as np
from pandas import DataFrame
import keras
import datetime
import dateutil
from dateutil.relativedelta import relativedelta
from sklearn import tree
import pydotplus
class DataTreatO:
    def __init__(self,df,rename_term,rename_damage):
        self.df=df
        self.rename_term=rename_term
        self.rename_damage=rename_damage
    def num2alpha(self,num):
        """ 数字numのアルファベットへの変換
            num>=26のときはアルファベット2文字となる
        """ 
        if num<=26:
            return chr(64+num)
        elif num%26==0:
            return num2alpha(num//26-1)+chr(90)
        else:
            return num2alpha(num//26)+chr(64+num%26)
    def DamagePick(self,damage):
        """ 損傷データのうち，損傷がTrueとなるデータを含んだ損傷名リストを戻す
        　　旧バージョンでは引数，DamageData,RenameDamage
        """
        #damage = pd.read_csv(DamageData)
        dcolumns=damage.columns
        #rename_damage=pd.read_csv(RenameDamage,header=None,dtype=str)
        d_damage=self.rename_damage.values
        n_damage=len(d_damage)
        num_data=len(damage)
        dam_data=[]
        for i in range(n_damage):
            dname=dcolumns[i]
            aa=damage[dname].value_counts()
            if aa[0]!=num_data:
                dam_data.append(self.rename_damage.iloc[i,1])
        return dam_data
    def Catego_Treat(self,row_name,train):
        """ Categorical項目に対する処理
        """
        t_Damage=DataFrame(['[-------    '+row_name],columns=['category'])
        t_Damage['ind']=['-------]']
        data=train[[row_name]]
        data=DataFrame('t_'+data[row_name]) #added 2020.6.27
        ll=data[row_name].unique()
        num=len(ll)
        l=list('')
        for i in range(num):
            if type(data[row_name][0])==np.bool_:
                #データがTrue,Falseで書かれている場合の処理
                for j in range(len(data)):
                    data[row_name][j]=int(data[row_name][j])
            else:
                data=data.replace({row_name:{ll[i]:i}})
            l.append(self.num2alpha(i+1))
        bb = DataFrame(ll, columns=['category']) #変更1
        bb['ind']=l #変更2
        t_Damage=t_Damage.append(bb)
        ### categorical process
        aa=np.array(data[row_name]) #変更3
        Damage=DataFrame(data=keras.utils.to_categorical(aa))#変更4
        for i in range(len(Damage.columns)): ##変更5
            Damage=Damage.rename(columns={i:row_name+self.num2alpha(i+1)})#変更6
        return t_Damage,Damage   
    def Numerical_Treat(self,row_name,dff):
        """ 数値項目に対する処理
        """
        num=len(dff)
        aa=np.zeros(num)
        nd=0
        dd=0.0
        for i in range(num):
            if dff[row_name][i]!='Null':  #Nullデータについてはとりあえず除外しておく
                aa[i]=dff[row_name][i]
                nd += 1
                dd += float(dff[row_name][i])
        uMean=dd/nd #Null以外の平均値
        for i in range(num):
            if dff[row_name][i]=='Null': #Nullデータは平均値に置き換える
                aa[i]=uMean
        Damage=DataFrame(aa,columns=[row_name])
        return Damage

    def DataTreat(self):
        """ UniPlannerデータに対するデータ処理ルーチン
        """
        df=self.df
        d_term=self.rename_term.values
        n_term=len(d_term)
        for i in range(n_term):
            df=df.rename(columns={d_term[i,0]:d_term[i,1]})
        d_damage=self.rename_damage.values
        n_damage=len(d_damage)
        for i in range(n_damage):
            df=df.rename(columns={d_damage[i,0]:d_damage[i,1]})
            #使用開始日からの経過日数を計算しuseD欄へ入力
        num=len(df)
        aa=np.zeros(num)
        jnum=0
        for i in range(num):
            if df.useDate[i]!='Null':  #Nullデータについてはとりあえず除外しておく
                bb=dateutil.parser.parse(str(df.riskDate[i]))-dateutil.parser.parse(str(df.useDate[i]))
                aa[i]=bb.days
                jnum += 1
        uMean=sum(aa)/jnum #Null以外の経過日数の平均値
        for i in range(num):
            if df.useDate[i]=='Null': #Nullデータは平均値に置き換える
                aa[i]=uMean
        useD=DataFrame(aa,columns=['useD']) 
        df=df.join(useD)
        # 不要項目の削除
        for i in range(n_term):
            if self.rename_term.iloc[i,2] == '-1':
                df=df.drop([self.rename_term.iloc[i,1]],axis=1)
        #DF->True or False
        dname=np.array(self.rename_damage.iloc[:,1]) #損傷名配列
        n_damage=len(dname)
        threshold=2
        damage=DataFrame(np.zeros(num),columns=['damage'])
        df=df.join(damage)
        #
        for i in range(n_damage):
            df.loc[df[dname[i]] < threshold, 'damage'] = 'False'
            df.loc[df[dname[i]] >= threshold, 'damage'] = 'True'
            df[dname[i]]=df.damage
        df=df.drop(['damage'], axis=1)
        #損傷機構部分の抽出
        damage=df.loc[:,dname]
        df=df.drop(damage,axis=1)
        num_term=len(df.columns)
        term=df.columns
        data=DataFrame(np.zeros(num),columns=['temp'])
        t_data=DataFrame(['(category'],columns=['category'])
        t_data['ind']=['list)']
        for i in range(num_term):
            if term[i]=='useD':
                categ=self.Numerical_Treat(term[i],df)
                data=data.join(categ)
                continue
            dd=self.rename_term[self.rename_term.iloc[:,1]==term[i]]
            if dd[2].values[0]=='0':
                categ=self.Numerical_Treat(term[i],df)
                data=data.join(categ)
                continue
            if dd[2].values[0]=='1':
                t_categ,categ=self.Catego_Treat(term[i],df)
                data=data.join(categ)
                t_data=t_data.append(t_categ)
        data=data.drop(['temp'],axis=1)
        return data,t_data,damage
    def Convert(self,InputData,RenameTerm,RenameDamage,OutputData,   CategoryList,DamageData):
        df = pd.read_csv(InputData)
        #項目名，損傷名の変数名への変換
        rename_term=pd.read_csv(RenameTerm,header=None,dtype=str)
        rename_damage=pd.read_csv(RenameDamage,header=None,dtype=str)
        data,t_data,damage=DataTreat(df,rename_term,rename_damage)
        data.to_csv(OutputData,index=False)
        damage.to_csv(DamageData,index=False)
        t_data.to_csv(CategoryList,index=False,encoding='cp932')
    def DamageTake(self,damage):
        """ 損傷データのうち，損傷がTrueとなるデータを含んだ損傷名リストを戻す
        入力:
            DataTreatO.DateTreatの出力のdamage
        処理結果:
            dam_data[]に，処理用の変数名
            damage_name[]に，UniPlannerでの損傷名が格納される
        出力:
            dam_data,damage_name
        """
        dcolumns=damage.columns
        d_damage=self.rename_damage.values
        n_damage=len(d_damage)
        num_data=len(damage)
        dam_data=[]
        damage_name=[]
        for i in range(n_damage):
            dname=dcolumns[i]
            aa=damage[dname].value_counts()
            if aa[0]!=num_data:
                dam_data.append(self.rename_damage.iloc[i,1])
                damage_name.append(self.rename_damage.iloc[i,0])
        return dam_data,damage_name
    def DrawTree(self,data,dtree):
        """ 決定木描画
        """
        mechanism=['False','True']
        dot_data = tree.export_graphviz(
            dtree, 
            out_file=None,
            feature_names=data.columns,
            class_names=mechanism,
            filled=True,
            proportion=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        return graph
def GetData(plant='A'):
    """
    目的:教師用サンプルデータの読み取り
    入力:
        sel='A' プラントAデータ
        sel='B' プラントBデータ
    出力:df,rename_term,rename_damage
        df              教師用サンプルデータ(DataFrame形式)
        rename_term     項目名変更テーブル
        rename_damage   損傷機構名変更テーブル 
    """
    base_dir = join(dirname(__file__), 'data/')
    if plant=='A':
        data_filename = join(base_dir, 'PlantA_Data.csv')
        RenameTerm = join(base_dir, 'rename_termA.csv')
        RenameDamage=join(base_dir, 'rename_damageA.csv')
    else:
        data_filename = join(base_dir, 'PlantB_Data.csv')
        RenameTerm = join(base_dir, 'rename_termB.csv')
        RenameDamage=join(base_dir, 'rename_damageB.csv')
    df=pd.read_csv(data_filename)
    rename_term=pd.read_csv(RenameTerm,header=None,dtype=str)
    rename_damage=pd.read_csv(RenameDamage,header=None,dtype=str)
    return df,rename_term,rename_damage
               