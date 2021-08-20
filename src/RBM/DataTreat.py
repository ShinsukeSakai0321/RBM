"""
  RBMへのAI適用にあたりデータ加工関連処理を行うパッケージ
  Copyright © 2021 Shinsuke Sakai, YNU. All Rights Reserved.
"""
from os.path import dirname, exists, expanduser, isdir, join, splitext
import pandas as pd
import numpy as np
from pandas import DataFrame
import keras
from keras.utils import to_categorical
import datetime
import dateutil
from dateutil.relativedelta import relativedelta
from sklearn import tree
import pydotplus
class DataTreatO:
    """
    Copyright © 2021 Shinsuke Sakai, YNU. All Rights Reserved.
    目的:
        UniPlannerの入力データをAIに利用するためのデータ処理
        このままでは、新規データに対して予測させる際に、カテゴリーデータ
        に対して、困難が伴うことが判明。このため、新たにclass DataTreatNを
        開発することとし、このclassは旧バージョンとしてとりあえず置いておく
    引数:
        df              オリジナルのUniPlannerデータ
        rename_term     項目名の変換テーブル
        rename_damage   損傷名の変換テーブル
        type            'AI_S' or 'AI_new' デフォールト'AI_S'
    """
    def __init__(self,df,rename_term,rename_damage,type='AI_S'):
        self.df=df
        self.rename_term=rename_term
        self.rename_damage=rename_damage
        self.type=type
    def num2alpha(self,num):
        """ 数字numのアルファベットへの変換
            num>=26のときはアルファベット2文字となる
        """ 
        if num<=26:
            return chr(64+num)
        elif num%26==0:
            return self.num2alpha(num//26-1)+chr(90)
        else:
            return self.num2alpha(num//26)+chr(64+num%26)
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
        Damage=DataFrame(data=to_categorical(aa))#変更4
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
        ### 以下の処理は、AI_Sに対してのみ。AI_newには行わない2021.2.10
        if self.type == 'AI_S':
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
        if self.type=='AI_S':
            threshold=2
        else:
            threshold=1
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


class DataTreatN:
    """
    Copyright © 2021 Shinsuke Sakai, YNU. All Rights Reserved.
    目的:
        UniPlannerの入力データをAIに利用するためのデータ処理
        このままでは、新規データに対して予測させる際に、カテゴリーデータ
        に対して、困難が伴うことが判明。このため、新たにclass DataTreatNを
        開発することとする
        Ver 1.0.1以降追加
    引数:
        df              オリジナルのUniPlannerデータ
        rename_term     項目名の変換テーブル
        rename_damage   損傷名の変換テーブル
        type            'AI_S' or 'AI_new' デフォールト'AI_new'
    """
    def __init__(self,df,rename_term,rename_damage,type='AI_new'):
        self.df=df
        self.rename_term=rename_term
        self.rename_damage=rename_damage
        self.type=type
    def CutDamage(self,damage,n_thres):
        """
        目的:データフレームdamageに記録される損傷モードのうち、
        　　　記録されるレコード数がn_thres以下の列を削除した
        　　　データフレームを返す
        """
        dam=damage.copy()
        d_lab=list(dam.columns)
        for i in range(len(d_lab)):
            damg=d_lab[i]
            if dam[damg].sum()<=n_thres:
                dam=dam.drop(damg,axis=1)
        return dam
    def num2alpha(self,num):
        """ 数字numのアルファベットへの変換
            num>=26のときはアルファベット2文字となる
        """ 
        if num<=26:
            return chr(64+num)
        elif num%26==0:
            return self.num2alpha(num//26-1)+chr(90)
        else:
            return self.num2alpha(num//26)+chr(64+num%26)
    def my_removeprefix(self,s, prefix):
        """
        sのプレフィックス（先頭の特定文字列）prefixを削除する
        """
        if s.startswith(prefix):
            return s[len(prefix):]
        else:
            return s
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
    def Catego_Treat(self,org_name,row_name,train): # org_name追加 2021.4.21
        """ Categorical項目に対する処理
            t_Damageの出力フォーマットの変更(2021.4.21)
            理由:新規データの読み込みのためには、カテゴリー項目の変換テーブルが必要になる
            　　DataFrameで利用しやすい形で出力するものとする。Damageについては変更しない
              　ものとする。
        """
        t_Damage=DataFrame(columns=['Columns','Category','newColumns']) #空のDataFrameの生成
        data=train[[row_name]]
        data=DataFrame('t_'+data[row_name]) #added 2020.6.27
        ll=data[row_name].unique()
        num=len(ll)
        l_org=list('')
        l=list('')
        for i in range(num):
            l_org.append(org_name)
            if type(data[row_name].iloc[0])==np.bool_: #<------2021.4.23 DataFrame[0]は行index=0の意味、ここでは第0行だからDataFrame.iloc[0]
                #データがTrue,Falseで書かれている場合の処理
                for j in range(len(data)):
                    data[row_name].iloc[j]=int(data[row_name].iloc[j])#<------2021.4.23 上に同じ
            else:
                data=data.replace({row_name:{ll[i]:i}})
            l.append(row_name+self.num2alpha(i+1))
        #bb = DataFrame(ll, columns=['category']) #変更<-----------------------
        #bb['ind']=l #変更2<-----------------------------

        bb=DataFrame(l_org,columns=['Columns'])
        bb['Category']=ll
        bb['newColumns']=l
        t_Damage=t_Damage.append(bb)
        ### categorical process
        aa=np.array(data[row_name]) #変更3
        Damage=DataFrame(data=to_categorical(aa))#変更4
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
            if dff[row_name].iloc[i]!='Null':  #Nullデータについてはとりあえず除外しておく,2021.4.23 DataFrame[0]は行index=0の意味、ここでは第0行だからDataFrame.iloc[0]
                aa[i]=dff[row_name].iloc[i]#<------2021.4.23 DataFrame[0]は行index=0の意味、ここでは第0行だからDataFrame.iloc[0]
                nd += 1
                dd += float(dff[row_name].iloc[i])#<------2021.4.23 DataFrame[0]は行index=0の意味、ここでは第0行だからDataFrame.iloc[0]
        uMean=dd/nd #Null以外の平均値
        for i in range(num):
            if dff[row_name].iloc[i]=='Null': #Nullデータは平均値に置き換える2021.4.23 DataFrame[0]は行index=0の意味、ここでは第0行だからDataFrame.iloc[0]
                aa[i]=uMean
        Damage=DataFrame(aa,columns=[row_name])
        return Damage

    def DataTreat(self):
        """ UniPlannerデータに対するデータ処理ルーチン
        """
        df=self.df
        df2=self.df #org_name保存 2021.4.21
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
        ### 以下の処理は、AI_Sに対してのみ。AI_newには行わない2021.2.10
        if self.type == 'AI_S':
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
                df2=df2.drop([self.rename_term.iloc[i,0]],axis=1) #2021.4.21
        #DF->True or False
        dname=np.array(self.rename_damage.iloc[:,1]) #損傷名配列
        n_damage=len(dname)
        """ DF計算値から損傷のTrue,Falseを判断するプロセスを無くしたので削除2021.4.26
        if self.type=='AI_S':
            threshold=2
        else:
            threshold=1
        damage=DataFrame(np.zeros(num),columns=['damage'])
        df=df.join(damage)
        #
        for i in range(n_damage):
            df.loc[df[dname[i]] < threshold, 'damage'] = 'False'
            df.loc[df[dname[i]] >= threshold, 'damage'] = 'True'
            df[dname[i]]=df.damage
        df=df.drop(['damage'], axis=1)
        """
        #損傷機構部分の抽出
        damage=df.loc[:,dname]
        df=df.drop(damage,axis=1)
        num_term=len(df.columns)
        term=df.columns
        data=DataFrame(np.zeros(num),columns=['temp'])
        #t_data=DataFrame(['(category'],columns=['category'])
        #t_data['ind']=['list)']
        t_data=DataFrame(columns=['Columns','Category','newColumns']) #空のDataFrameの生成
        org_name=df2.columns #2021.4.21
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
                t_categ,categ=self.Catego_Treat(org_name[i],term[i],df)
                data=data.join(categ)
                t_data=t_data.append(t_categ)
        data=data.drop(['temp'],axis=1)
        #Category名の先頭の't_'を削除
        aa=t_data['Category']
        for i in range(len(aa)):
            aa.iloc[i]=self.my_removeprefix(aa.iloc[i],'t_')
        t_data['Category']=aa
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
class DataTreatNew:
    """
    Copyright © 2021 Shinsuke Sakai, YNU. All Rights Reserved.
    目的:　新規データに対するデータ加工
    Ver. 1.1.1以降追加
    入力:
        new_file:         新規のUniPlanner入力データファイル名(Default:'df_new_data.csv'),損傷モードデータは含まない
        rename_term_file: 入力データの項目変換テーブルファイル名(Default:'rename_term.csv')
        t_data_file:      学習時に吐き出されたt_dataファイル名(Default:'t_data.csv')
        data_file:        学習時に吐き出されたdataファイル名(Default:'data_train.csv')
    """
    def __init__(self,new_file='df_new_data.csv',rename_term_file='rename_term.csv',
                 t_data_file='t_data.csv',data_file='data_train.csv'):
        df_new=pd.read_csv(new_file,index_col=0)#新規データ
        t_data=pd.read_csv(t_data_file, index_col=0)
        rename_term=pd.read_csv(rename_term_file,header=None,dtype=str)
        data=pd.read_csv(data_file,index_col=0)#学習時に作ったdataを読み取る
        self.df=df_new
        self.rename_term=rename_term
        self.t_data=t_data
        self.colData=data.columns#その列名を取得
        #空のデータフレーム
        self.dataNew = pd.DataFrame(index=[], columns=self.colData)
    def rename(self):
        """
        目的:　新規データのcolumnsについて、rename_termに基づくリネーム
        """
        d_term=self.rename_term.values
        n_term=len(d_term)
        #項目名のリネーム
        for i in range(n_term):
            self.df=self.df.rename(columns={d_term[i,1]:d_term[i,1]})
        # 不要項目の削除
        for i in range(n_term):
            if self.rename_term.iloc[i,2] == '-1':
                self.df=self.df.drop([self.rename_term.iloc[i,0]],axis=1)
    def DataConvert1(self,dlist):
        """
        目的:　1行分の新規データについて変換後データを戻す
        入力:
            dlist:    一行分データのDataFrame
        """
        dCol=self.df.columns
        num_term=len(dCol)
        dNew = pd.DataFrame(index=[], columns=self.colData)
        #全データとして0を代入
        l=[0]*len(self.colData)
        dNew.loc[0]=l
        for i in range(num_term):
            dd=self.rename_term[self.rename_term.iloc[:,0]==dCol[i]]
            if dd[2].values[0]=='0':  #numerical term
                self.dataNew[dd[1].values[0]]=dlist[i]
            elif dd[2].values[0]=='1': #categorical term
                gg=self.t_data[(self.t_data['Columns']==dd[0].values[0]) & (self.t_data['Category']==dlist[i]) ]
                if gg.empty != True:  #当初のカテゴリーに含まれていないデータはスキップ
                    dNew[gg['newColumns'].iloc[0]]=1
        return dNew
    def DataConvert(self):
        """
        目的:　新規データの解析用データへの変換
        """
        for i in range(len(self.df)):
            dlist=self.df.iloc[i] #i番目のデータ取得
            self.dataNew=self.dataNew.append(self.DataConvert1(dlist))
        return self.dataNew
    def GetConvertedData(self):
        """
        目的:　変換後のデータの取得
        """
        return self.dataNew
    def EvalNull(self,data,ratio): 
        """
        目的:　リストdataの中で、'Null'の比率がratio以上のときtrue
                2021.5.18
        """
        return sum(data=='Null')/len(data)>ratio
    def CutNull(self,categorical_features,ratio):
        """
        目的:　リストcategorical_featuresの中で、'Null'の比率がratio以上の項目を削除後、リストを返す
                2021.5.18
        """
        cat=categorical_features.copy()
        for categ in zip(categorical_features):
            if self.EvalNull(self.df[categ[0]],ratio):
                cat.remove(categ[0])
        return cat
class DamageTreat:
    """
    Copyright © 2021 Shinsuke Sakai, YNU. All Rights Reserved.
    目的:全損傷モードについて、ラベル化するための管理プログラム
    Ver. 1.0.2以降追加
    利用法:
        下記にてインスタンスを生成する
        ins=DamageTreat(data,damage)
    引数:
        data    DataTreatN::DataTreat()によって加工されたUniPlannerデータの入力データ項目部分
        damage  DataTreatN::DataTreat()によって加工されたUniPlannerデータの損傷データ部分
    """
    def __init__(self,data,damage):
        self.data=data
        self.damage=damage
    def MakeDamage(self):
        """
        目的:全損傷データをラベル化するためのデータ加工
        　　　各入力レコードごとに、損傷モードを番号としてラベル化しtargetに入力する
           　 番号は、DM***の損傷名に対して***の部分を割り当てる。同一レコードに対して
              複数の損傷モードが存在するときには、その数だけ入力レコードをコピーし、各
              損傷モードのラベルをtargetに入力する
        入力:なし
        出力:damageNew,dam_list
              damageNew:  加工後の学習用入力データ
              dam_list:   加工後のtargetデータ
        """
        col=self.damage.columns.values
        cols=self.data.columns
        dam_list=[]
        self.damageNew=pd.DataFrame(index=[], columns=cols)#新たに作り直す学習用データ
        for i in range(len(self.data)):
            aa=self.damage.iloc[i]==1
            dam=col[aa]#損傷名リスト
            # 損傷モードが0のときには、0として記録しておく
            if len(dam)==0:
                self.damageNew=self.damageNew.append(self.data.iloc[i])
                dam_list.append(0)
            for j in range(len(dam)):
                self.damageNew=self.damageNew.append(self.data.iloc[i])
                nn=int(dam[j].replace('DM',''))
                dam_list.append(nn)
        return self.damageNew,dam_list

class DamageAnal:
    """
    Copyright © 2021 Shinsuke Sakai, YNU. All Rights Reserved.
    目的:　決定木解析結果に対する評価
    Ver. 1.0.5以降追加
    """
    def __init__(self,dtree):
        self.dtree=dtree
    def PredictDmode(self,data):
        """
        目的:  新規データに対する予測を行い、各レコードの損傷モードとそ　　　　の確率からなるJsonデータを返す
        入力:  data  予測するための新規データ
        出力:  dam_and_prob  損傷モードと損傷モード確率からなるJson
                            データ
        """
        self.data=data
        tt=self.dtree.predict_proba(data)
        self.proba=pd.DataFrame(tt)#予測確率値のデータフレーム
        col_class=self.dtree.classes_ #probaの列名に出てくるクラス名の一覧
        self.proba.columns=col_class #クラス名をその番号に書き換える
        dam_and_prob=[]
        col_lab=self.proba.columns
        col_num=len(col_lab)
        for i in range(len(data)):
            aa=self.proba.iloc[i]
            dam=[] #予測された損傷モード
            prob=[] #損傷モードの予測確率値
            for j in range(col_num):
                pp=aa[col_lab[j]]
                if pp != 0.0:
                    dam.append(col_lab[j])
                    prob.append(pp)
            # 損傷モードなし(ラベル0)を他の損傷モードとともに予測した場合は、ラベル0は削除する
            if dam[0]==0 and len(dam)>1:
                dam.pop(0)
                prob.pop(0)
            t_data= {'record':i,'damage':dam,'probability':prob}
            dam_and_prob.append(t_data)
            #self.dam_and_prob=dam_and_prob
        return dam_and_prob
    def damByProb(self,thres,dam_and_prob,MaxNorm=True):
        """
        目的:  直前にPredictDmodeで評価された各レコードの損傷モード確率
            に基づき、閾値をthresとする損傷モードの抽出を行いJson
            データとして返す
        入力:  thres  損傷モード抽出のための確率値の閾値
            dam_and_prob PredictDmodeの出力
            MaxNorm  Trueのとき、各レコードの損傷モード確率の最大値で、確率値を基準化する
            　　　　default値はTrue
        出力:  dam_by_prob  抽出された損傷モードのJsonデータ
        """
        dam_by_prob=[]
        for i in range(len(dam_and_prob)):
            prob=dam_and_prob[i]['probability']
            dam=dam_and_prob[i]['damage']
            dam_picked=[]
            prob_picked=[]
            if MaxNorm:  #確率値の最大値で基準化
                p_max=max(prob)
                prob=prob/p_max
            for j in range(len(prob)):
                pp=prob[j]
                if pp > thres:
                    dam_picked.append(dam[j])
                    prob_picked.append(prob[j])
            t_data= {'record':i,'damage':dam_picked,'probability':prob_picked}
            dam_by_prob.append(t_data)
        return dam_by_prob
    def toJson(self,damage,dam_by_prob):
        """
        目的:決定木解析で得られたレコードごとのラベルに対する確率値を予測損傷モード、正解損傷モードとともにJson化する
        入力: damage   損傷モードを記録するデータフレーム
              dam_by_prob  damByProbの出力
        出力: df   record,data,damage,probabilityのJsonデータ
        """
        #self.damage=damage
        df={}
        col_lab=self.proba.columns
        col_num=len(col_lab)
        ans_col_lab=damage.columns
        ans_col_num=len(ans_col_lab)
        for i in range(len(dam_by_prob)):
            dam=dam_by_prob[i]['damage']
            prob=dam_by_prob[i]['probability']
            dd=damage.iloc[i]
            ans=[]
            for j in range(ans_col_num):
                ii=dd[j]
                if ii==1:
                    na=ans_col_lab[j]
                    dt=int(na.replace('DM',''))
                    ans.append(dt)
            if len(ans)==0:
                ans.append(0)
            t_data= {"data":ans,"damage":dam,"probability":prob}
            df[str(i)]=t_data
        self.df=df
        return df
    def toJson2(self,dam_by_prob):
        """
        目的:決定木解析で得られたレコードごとのラベルに対する確率値を予測損傷モードとともにJson化する
        入力: dam_by_prob  damByProbの出力
        出力: df   record,damage,probabilityのJsonデータ
        """
        #self.damage=damage
        df={}
        col_lab=self.proba.columns
        col_num=len(col_lab)
        for i in range(len(dam_by_prob)):
            dam=dam_by_prob[i]['damage']
            prob=dam_by_prob[i]['probability']
            t_data= {"damage":dam,"probability":prob}
            df[str(i)]=t_data
        return df
    def checkMatch(self,df,defThres=0.75):
        """
        目的:決定木解析の確率値に基づくマッチング評価
        入力:
             df:  toJsonの出力値
             defThres:  包含条件として、(予測-正解)/予測<=defThres
             　　　　　　とする(Default:0.75) v1.2.2で追加
        出力:
             cE:     評価結果がtargetの損傷モード内容と完全一致したレコード数
             cP:     targetの損傷モード内容と完全一致はしないが、包含しているレコード数
        """
        #精度検証
        #thres=0.3# 確率値の打ち切り閾値
        cE=0
        cP=0
        for i in range(len(df)):
            data=df[i]['data']#正解
            damage=df[i]['damage']#予測結果
            prob=df[i]['probability']
            dam_pick=[]

            if data==damage:
                cE += 1
            if data < damage:
                sabun=list(set(damage)-set(data))#差分のリスト
                rat=len(sabun)/len(damage)
                if rat<=defThres:#差分の割合がdefThres以下のとき包含と判定
                    cP += 1        
        return cE,cP
    def FinalRes(self,dam_by_prob,dam_name='damage_name.csv'):
        """
        目的:推論により得られたdamage_by_probから、各レコードの損傷モードを確率値つきで、書式を整えて出力する
        v.1.1.3以降追加
        入力:  dam_by_prob     DamageAnal::damByProbにより処理された結果
            　 dam_name        損傷番号と損傷名の対応を記載したcsvファイル名(default damage_name.csv)
        出力　各レコードについて予測される損傷モードを記載したDataFrame
                出力フォーマットは下記の通り
                {損傷名1(p1)}{損傷名2(p2)}....
                .
                .
                
                p1,p2は確率値を示す。ただし、最大値が1となるように基準化している。
        """
        damage_name=pd.read_csv(dam_name)
        drec=[]
        for i in range(len(dam_by_prob)):
            dd=dam_by_prob[i]
            d1=dd['damage']
            srec=''
            for j in range(len(d1)):
                cc=d1[j]
                if cc==0:
                    dam='なし'
                else:
                    dam=damage_name['damage'][cc-1]
                srec += '{'+dam +'('+str(dd['probability'][j])+')}'
            drec.append(srec)
        return pd.DataFrame({'DamageMode':drec})
    def ResultFrame(self,dam_by_prob,dam_name='damage_name.csv'):
        """
        目的:推論により得られたdamage_by_probから、各レコードの損傷モードテーブルをDataFrameで戻す
                v.1.1.3以降追加
        入力:  dam_by_prob     DamageAnal::damByProbにより処理された結果
            　 dam_name        損傷番号と損傷名の対応を記載したcsvファイル名(default damage_name.csv)
        出力:　各レコードについて予測される損傷モードに1のフラグを立てたテーブルをDataFrameとしたもの
        """
        damage_name=pd.read_csv(dam_name)
        nd=len(damage_name)
        l=[0]*len(dam_by_prob)
        res=pd.DataFrame()
        for i in range(nd):
            col='DM'+str(i+1)
            res[col]=l
        for i in range(len(dam_by_prob)):
            dd=dam_by_prob[i]
            d1=dd['damage']
            for j in range(len(d1)):
                if d1[j]!=0:
                    dm='DM'+str(d1[j])
                    res[dm][i]=1
        return res
from sklearn import tree
class GeneralTrain:
    """
    目的:RBMデータのデータ処理から、学習までのプロセスの汎用処理サポート
    入力:
        fname:         UniPlannerの入力データ名
        r_term:        rename_termのファイル名(Default:'rename_term.csv')
        r_damage:      rename_damageのファイル名(Default:'rename_damage.csv')
    """
    def __init__(self,fname,r_term='rename_term.csv',r_damage='rename_damage.csv'):
        self.df=pd.read_csv(fname)
        self.rename_term=pd.read_csv(r_term,header=None,dtype=str)
        self.rename_damage=pd.read_csv(r_damage,header=None,dtype=str)
    def DataTreat(self,n_thres=10):
        """
        目的:RBMデータのデータ処理
        入力:
            n_thres:    損傷モードのレコード数の閾値(Default:10)
        結果:
            self.damage_cut    閾値以上のレコード数の損傷モードDataFrame
            self.data_new      学習に用いる入力DataFrame
            self.target        学習に用いる損傷モードのtarget
        出力:
            data        入力データの加工前の学習用データ
        """
        #DataTreat処理
        # インスタンス生成
        dtoData=DataTreatN(self.df,self.rename_term,self.rename_damage,type='AI_new')
        # データ加工
        data,self.t_data,damage=dtoData.DataTreat()#data:学習用に加工された入力データ、damage:損傷データ
        # 登録されている損傷モードのレコード数がn_thres以下の損傷モードを削除する
        self.damage_cut=dtoData.CutDamage(damage,n_thres)
        aa=DamageTreat(data,self.damage_cut)                           #################dt削除のこと
        self.data_new,self.target=aa.MakeDamage()
        return data
    def GetDamage(self):
        """
        目的:DataTreatの処理後に、n_thres以上の損傷モードが抽出された損傷モードデータを抽出する
        """
        return self.damage_cut
    def DecisionTree(self,max_depth=50):
        """
        目的:決定木による機械学習
        入力:
            max_depth:     決定木解析の深さ(Default:50)
        結果:
            self.dtree     決定木解析の結果ファイル
        """

        max_depth=int(max_depth)
        self.dtree = tree.DecisionTreeClassifier(max_depth=max_depth,random_state=0)
        self.dtree = self.dtree.fit(self.data_new, self.target)
    def DAnal(self,data,dtree,thres=0.2):
        """
        目的:決定木解析結果について、確率値処理のためのデータ加工および精度チェックを行う
        入力:
            data        決定木解析結果の推論エンジンを用い、dataに対する予測を行う
            dtree  決定木解析の結果
            thres       損傷と判断するための基準化確率値の閾値(Default:0.2)
        結果:
            self.pJson      正解と予測結果を合体したJsonデータ
            self.cE      損傷モードが完全一致した数
            self.cP      損傷モードが包含一致した数
        出力:
            dam_by_prob  基準化確率値の閾値による抽出された損傷モード
            fullRate     完全一致率
            PartialRate  包含率
        """
        da=DamageAnal(dtree)#解析用のインスタンス生成                         ######### dt削除のこと
        dam_and_prob=da.PredictDmode(data)#各レコードの予測損傷モード確率の取り出し
        dam_by_prob=da.damByProb(thres,dam_and_prob)#確率値の閾値による抽出
        self.pJson=da.toJson(self.damage_cut,dam_by_prob)#正解と予測結果を合体したJsonデータ作成 
        # 一致率のチェック
        self.cE,self.cP=da.checkMatch(self.pJson)# 一致率のチェック
        num=len(data)
        fullRate=self.cE/num
        partialRate=(self.cE+self.cP)/num
        return dam_by_prob,fullRate,partialRate
        #print('完全一致率=',cE/num,'包含率=',(cE+cP)/num)
        #完全一致率= 0.9337899543378996 包含率= 0.984779299847793
    def TrainTest(self,ratio=0.2,thres=0.2,max_depth=50,n_thres=10):
        """
        目的:RBMデータに対し、過学習のチェックをするたtrain,testに対する検証を行う
        入力:
            ratio:         fnameの入力データについてtrain:testの割合を1:ratioにする
            　　　　　　　　　(Default:0.2)
            thres:         損傷モード判定のための基準化確率値閾値(Default:0.2)
            max_depth:     決定木解析の深さ(Default:50)
        結果:
            self.dtree     決定木解析の結果ファイル
        出力:
            self.dtree     決定木解析の結果ファイル
            df_train       trainの学習用に用いた入力データ
            t_data         変換テーブル表
            res            完全一致率と包含率の評価結果をtrainとtestについてDataFrameにした結果
        """
        # 全体データに対する加工処理
        data=self.DataTreat(n_thres=n_thres)
        #過学習に対する検討
        # dataとdamage_cutに対して、ratioの割合でtestを切り出し、残りをtrainとする
        #まずはdataとdamage_cutを合体する
        df=data.join(self.damage_cut)
        df_test=df.sample(frac=ratio)#ratioの割合でtestデータの切り出し
        df_train=df.drop(df_test.index)#残りをtrainとする
        df_train_data=df_train.drop(self.damage_cut.columns,axis=1)#trainデータ部分の切り出し
        df_train_damage=df_train.drop(data.columns,axis=1)#train損傷モード部分の切り出し
        df_test_data=df_test.drop(self.damage_cut.columns,axis=1)#testデータ部分の切り出し
        df_test_damage=df_test.drop(data.columns,axis=1)#test損傷モード部分の切り出し
        a_train=DamageTreat(df_train_data,df_train_damage)     #############dt部分削除のこと
        X_train,y_train=a_train.MakeDamage()
        # trainに対する決定木解析
        max_depth=int(max_depth)
        dtree = tree.DecisionTreeClassifier(max_depth=max_depth,random_state=0)
        self.dtree = dtree.fit(X_train, y_train)
        da=DamageAnal(self.dtree)#解析用のインスタンス生成                    dt部分削除のこと
        #for train data
        dam_and_prob=da.PredictDmode(df_train_data)#各レコードの予測損傷モード確率の取り出し
        dam_by_prob=da.damByProb(thres,dam_and_prob)#確率値の閾値による抽出
        dff=da.toJson(df_train_damage,dam_by_prob)#正解と予測結果を合体したJsonデータ作成
        cE,cP=da.checkMatch(dff)# 一致率のチェック
        num=len(df_train_data)
        #print('train:完全一致率=',cE/num,'包含率=',(cE+cP)/num)
        train_perfect=cE/num
        train_include=(cE+cP)/num
        if ratio!=0:
            #for test data
            dam_and_prob=da.PredictDmode(df_test_data)#各レコードの予測損傷モード確率の取り出し
            dam_by_prob=da.damByProb(thres,dam_and_prob)#確率値の閾値による抽出
            dff=da.toJson(df_test_damage,dam_by_prob)#正解と予測結果を合体したJsonデータ作成
            cE,cP=da.checkMatch(dff)# 一致率のチェック
            num=len(df_test_data)
            #print('test:完全一致率=',cE/num,'包含率=',(cE+cP)/num)
            test_perfect=cE/num
            test_include=(cE+cP)/num
        res=pd.DataFrame({'train':[train_perfect,train_include],'test':[test_perfect,test_include]},index=['完全一致率','包含率'])
        #完全一致率= 0.5551330798479087 包含率= 0.752851711026616
        return self.dtree,df_train_data,self.t_data,res