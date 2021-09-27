import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
class PreAnal:
    """
    目的:数値データ、カテゴリーデータが混在する入力データに対して感度分析を実施する
    利用法　入力データdfはdataframeとし、以下によりインスタンスを発生する
        sa = PreAnal(df)
    """
    def __init__(self,data):
        """
        dataはDataFrame
        """
        self.df=data
    def num2alpha(self,num):
        """
        数字のアルファベットへの変換
        """
        if num<=26:
            return chr(64+num)
        elif num%26==0:
            return self.num2alpha(num//26-1)+chr(90)
        else:
            return self.num2alpha(num//26)+chr(64+num%26)
    def SetCateg(self,categorical_features):
        self.categorical_features=categorical_features
        categorical_transformer = OneHotEncoder(handle_unknown='ignore',sparse=False)
        pre_cat = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)])
        clf_cat = Pipeline(steps=[('pre_cat', pre_cat)])
        aa_cat=clf_cat.fit_transform(self.df)
        tt_cat=pd.DataFrame(aa_cat)
        categorical_transformer = OneHotEncoder(handle_unknown='ignore',sparse=False)
        col_cat=[] #カテゴリー名のリスト
        self.data_cat=pd.DataFrame([]) #カテゴリーテーブル
        for name in zip(categorical_features):
            dd=self.df[name[0]].value_counts()
            aa=dd.index
            aa_contents=sorted(aa)
            cont=categorical_transformer.fit_transform(pd.DataFrame(self.df[name[0]]))
            aa_index=[]
            for i in range(cont.shape[1]):
                aa_index.append(name[0]+'_'+self.num2alpha(i+1))
                col_cat.append(name[0]+'_'+self.num2alpha(i+1))
            #データにスペースなどが記述されているときの対応 otherと記述される
            for i in range(cont.shape[1]-len(aa_contents)):
                aa_contents.append('other')
            self.data_cat=pd.concat([self.data_cat,pd.DataFrame({'index':aa_index,'contents':aa_contents})])
        tt_cat.columns=col_cat
        return tt_cat
    def SetNum(self,numeric_features):
        """dfの数値項目numeric_featuresに対して次の処置を行う
        　　・欠損値を数値データのmedian値で置き換える
        　　・平均値，標準偏差で基準化
        　　・ラベル名nemeric_features，整形データのDataFrameを戻す
            Pipelineの使い方については次の書籍を参照のこと
            「scikit-learn,Keras,TensorFlowによる実践機械学習」
            O'REILLY, 2.5.5 変換パイプライン
            **注意点**
            欠損値は，np.NaNであることを前提としている．'Null'などと記載されて
            いるときには，self.NullNaNconvertを使って事前に変換しておくこと
        """
        self.numeric_features=numeric_features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),#欠損値処理
            ('scaler', StandardScaler())])#平均値，標準偏差での基準化    
        pre_num = ColumnTransformer( #numeric_featuresのラベル列のみに適用
            transformers=[
                ('num', numeric_transformer, numeric_features)])
        clf_num = Pipeline(steps=[('pre_num', pre_num)])
        clf=clf_num.fit(self.df)
        self.clf_fit=clf
        aa_num=clf.transform(self.df)
        tt_num=pd.DataFrame(aa_num)
        col_num=[]
        for name in zip(numeric_features):
            col_num.append(name[0])
        tt_num.columns=col_num
        return tt_num
    def GetNumericClf(self):
        """数値項目について、欠損値処理、平均値・標準偏差基準化処理後のエンジンを戻す
        　 テストデータに対して処理するときは以下を実行すればよい
          　　clf_fit=pa.GetNumericClf()
              X_test_transformed = clf_fit.transform(X_test)
        """
        return self.clf_fit
    def GetCategTable(self):
        return self.data_cat
    def TermName(self,features):
        """featuresで与える入力項目記号リストをカテゴリー名リストに変換して返す
        """
        tab=self.GetCategTable()
        n_term=[]
        for pp in features:
            aa=list(tab[tab['index']==pp]['contents'])[0]
            n_term.append(aa)
        return n_term
    def Xmake(self,numeric_features,categorical_features):
        """
        数値項目とカテゴリー項目を結合して入力データのDataFrameを戻す
        """
        tt_num=self.SetNum(numeric_features)
        tt_cat=self.SetCateg(categorical_features)
        X=pd.concat([tt_num,tt_cat],axis=1)
        return X
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
    def NullNaNconvert(self,given_columns):
        """登録されているself.dfについて，given_columnsで指定されるコラム名のデータのうち，'Null'をNaNに書き換える
        """
        dd=self.df[given_columns].replace('Null',np.NaN)
        self.df=self.df.drop(given_columns,axis=1)
        self.df=self.df.join(dd)
    def DataTreatForTest(self,df_test,org_columns,categorical_features):
        #新規データdf_testのうちカテゴリカル項目についてorg_columnsと、GetCategTable()に従ってカテゴリカル処理する
        new_data=pd.DataFrame(columns=org_columns)
        table=self.GetCategTable()        
        df_categ=df_test[categorical_features]

        for i in range(len(df_test)):
            d1={}
            for term in org_columns:
                d1[term]=0
            dd=df_categ[i:i+1]
            col=dd.columns
            for cat in col:
                dat=dd[cat][i]
                picked=table[table["index"].str.startswith(cat)]
                target=picked[picked['contents']==dat]
                if len(target)!=0:# train時のカテゴリー項目にない項目のばあいスキップする
                    d1[target.iloc[0]['index']]=1
            new_data=new_data.append(d1, ignore_index=True)        
        return new_data