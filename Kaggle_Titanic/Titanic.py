# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 08:04:06 2017

@author: qiu
"""

    
import pandas as pd
import numpy as np
import math
from pandas import Series,DataFrame
data_train = pd.read_csv("train.csv")
##获取样本名称含有特定子名称的的数据集
Major=data_train[data_train['Name'].str.contains("Major")]
print(Major)


def read_data(df):   
    print(df.head(5))
    #print(df.tail(24))
    #print(df.dtypes)
    print(df.count())
    print(df.shape)
    print(df.describe())
    #print(df.columns)
    print(df.info())
    cols = df.columns.values[[1,2,3,4]]
    print (cols)
    for c in cols:
        print (c, df[c].unique())
#read_data(data_train)

##analysis_data
##略
##processing
##有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。
##年龄缺失采用拟合法
from sklearn.ensemble import RandomForestRegressor
def set_Dring_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
# 乘客分成已知年龄和未知年龄两部分.as_matrix()转换成矩阵类型不再是dataframe
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y=known_age[:,0]
    x=known_age[:,1:]
    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=1)
    rfr.fit(x,y)
    predictAge=rfr.predict(unknown_age[:,1:])
    df.loc[(df.Age.isnull()),'Age']=predictAge
    return df,rfr
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin']="Yes"
##修改原DataFrame的值    
    df.loc[(df.Cabin.isnull()),'Cabin']="No"
    return df
data_train,rfr=set_Dring_ages(data_train)
data_train = set_Cabin_type(data_train)
data_train
##我们通常会先对类目型的特征因子化/one-hot编码,变成数值型后再处理。
##其中一个想法是将一个属性多种类别，转换成多个属性，每个属性变为0-1类
##使用get_dummies进行one-hot编码
dummies_Cabin=pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked=pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex=pd.get_dummies(data_train['Sex'],prefix='Sex')    
dummies_Pclass=pd.get_dummies(data_train['Pclass'],prefix='Pclass')
data_train=pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df=data_train.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis=1)

##预处理——归一化处理
##https://sanwen8.cn/p/135RpHk.html
##TypeError: ‘NoneType’ object is not subscriptable
##错误原因可能是该变量类型有问题
import sklearn.preprocessing as preprocessing 
scaler=preprocessing.StandardScaler()
age_scale_param=scaler.fit(df['Age'])
df['Age_scaled']=scaler.fit_transform(df['Age'],age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'],fare_scale_param)
df
##产生数据集并训练
from sklearn import linear_model
data_np=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
data_np=data_np.as_matrix()#格式转换成array
x=data_np[:,1:]
y=data_np[:,0]
clf=linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf
clf.fit(x,y)
x.shape
##预处理测试集
data_test = pd.read_csv("test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)
df_test

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
prediction=predictions.astype(np.int64)
###将预测结果写成数据框的形式并存为csv格式
#result=pd.DataFrame({'Passenerld':data_test['PassengerId'].as_matrix(),'Survived':prediction})
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
#result.to_csv("logistic_regression_predictions.csv", index=False)
##看看学习情况(画学习曲线)

from sklearn.model_selection import learning_curve,ShuffleSplit
import matplotlib.pyplot as plt
def plot_learning_curve(estimator,title,x,y,ylim=None, cv=None,train_sizes=np.linspace(.1, 1.0, 20), n_jobs=1):
    train_sizes,train_scores,test_scores=learning_curve(estimator,x,y,train_sizes=np.linspace(.1, 1.0, 20))
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean=np.mean(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,color='r',alpha=0.1)
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,color='g',alpha=0.1)
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label="训练集上得分")  
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label="交叉验证集上得分")
    plt.legend(loc="best")
    return plt
title="Learning Curves (linear_model, C=1)"
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
plot_learning_curve(clf,title,x,y,train_sizes=np.linspace(.1, 1.0, 20),cv=cv, n_jobs=1)
plt.show()

###目前的曲线看来，我们的model并不处于overfitting的状态
###(overfitting的表现一般是训练集上得分高，而交叉验证集上要低很多，中间的gap比较大)。
###因此我们可以再做些feature engineering的工作，添加一些新产出的特征或者组合特征到模型中。
####方法：看看现在得到的模型的系数，因为系数和它们最终的判定能力强弱是正相关的
####（在逻辑回归中有效）
#print(list(test.columns))
#print(list(clf.coef_))
##print(pd.DataFrame({"columns":list(test.columns)[1:], "coef":list(clf.coef_.T)}))
####交叉验证
#from sklearn.model_selection import cross_val_score,train_test_split
##看看打分情况
#print(cross_val_score(clf,x,y,cv=5))
##交叉验证——分割数据
#split_train,split_cv=train_test_split(data_np,test_size=0.3,random_state=0)
#clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
#clf.fit(split_train[:,1:],split_train[:,0])
#交叉验证——预测数据(数据格式问题，暂时放一放)
#predicions=clf.predict(split_cv[:,1:])
##格式转换http://www.168seo.cn/python/1882.html
#split_cv=pd.DataFrame(split_cv)
#split_cv[predictions!=split_cv.as_matrix()[:,0]].drop()
### 去除预测错误的case看原始dataframe数据
#origin_data_train=pd.read_csv('Train.csv')
#bad_cases=origin_data_train['PassengerId'].isin(split_cv[split_cv[:,0]!=predictions]['PassengerId'].values)
#print(bad_cases)

##再将特征细化，深度挖掘
#1、Pclass和Sex俩太重要了，用它们去组出一个组合属性来，这也是另外一种程度的细化。
#总结：分析数据后，可以将两个或多个对结果影响将达的属性做个组合
data_train=pd.read_csv('Train.csv')
data_train['Sex_Pclass']=data_train.Sex+'_'+data_train.Pclass.map(str)
#print(data_train['Sex_Pclass'])
###按Mr，Miss，Mrs，Ms的均值分别填充缺省值
'''
##获取样本名称含有特定子名称的的数据集
Mr=data_train[data_train['Name'].str.contains('Mr')]
Mr_Age=round(np.mean(Mr.Age[Mr.Age.notnull()]))
Mr_Age

Mrs=data_train[data_train['Name'].str.contains('Mrs')]
Mrs_Age=round(np.mean(Mrs.Age[Mrs.Age.notnull()]))
Mrs_Age
Miss=data_train[data_train['Name'].str.contains('Miss')]
Miss_Age=round(np.mean(Miss.Age[Miss.Age.notnull()]))
Miss_Age
Ms=data_train[data_train['Name'].str.contains('Ms')]
Ms_Age=round(np.mean(Ms.Age[Ms.Age.notnull()]))
Ms_Age
'''
def find(arr,s):
    return [ x for x in arr if s in x]

x=find(data_train.Name,'x.')#找到含有x的名字
def loc_Age_nan(arr,x):
    x_age=[]   
    x_ages_index=[] 
    for i in range(len(list(x))):
        ages=arr.Age[arr.Name==x[i]].values#找到此名字对应的年龄和索引
        index_ages=arr.Age[arr.Name==x[i]].index
        if math.isnan(ages)==False:#要求是float格式，排除年龄这一栏有‘NAN’
            x_age.append(ages)
        if math.isnan(ages)==True:#要求是float格式
            x_ages_index.append(index_ages)#找到年龄‘nan’的索引
    x_ages_index
#print(x_ages_index)
    x_age=int(round(np.mean(x_age)))#求‘x’年龄的均值
    for j in range(len(list(x_ages_index))):
#http://blog.csdn.net/alanguoo/article/details/52331901
#对数据修改
        index_x=x_ages_index[j]
        index_x=list(index_x)
#将均值赋给‘nan’的地方
        arr.loc[arr.Age.index==index_x,'Age' ] = x_age
    return arr
#通过类型的均值填充缺省的位置
Mr=find(data_train.Name,'Mr.')#找到含有x的名字
Mr_age=[]
Mr_ages_index=[]
Mrs=find(data_train.Name,'Mrs.')#找到含有x的名字
Mrs_age=[]
Mrs_ages_index=[]
Miss=find(data_train.Name,'Miss.')#找到含有x的名字
Miss_age=[]
Miss_ages_index=[]
Master=find(data_train.Name,'Master.')#找到含有x的名字
Master_age=[]
Master_ages_index=[]
Dr=find(data_train.Name,'Dr.')#找到含有x的名字
Dr_age=[]
Dr_ages_index=[]
data_train=loc_Age_nan(data_train,Mr)
data_train=loc_Age_nan(data_train,Mrs)
data_train=loc_Age_nan(data_train,Miss)
data_train=loc_Age_nan(data_train,Master)
data_train=loc_Age_nan(data_train,Dr)
##单加child和mother
data_train['child']=data_train.Age[data_train.Age.values < 12]
data_train.loc[data_train.child.notnull(),'child']=1
data_train.loc[data_train.child.isnull(),'child']=0
               
Mrs=find(data_train.Name,'Mrs.')#找到含有x的名字
print(Mrs)
name_mrs=[]

data_train['mother']=0
for k in range(len(list(Mrs))):
    name_mrs=(data_train.Name[data_train.Name==Mrs[k]].index)
    name_mrs=list(name_mrs)
    print(name_mrs)
#IndexError: too many indices for array
#数组索引太多
    data_train.loc[data_train.mother.index==name_mrs,'mother'] = 1
data_train.mother[data_train.Parch<=1]=0
##特征
data_train = set_Cabin_type(data_train)
data_train
##我们通常会先对类目型的特征因子化/one-hot编码,变成数值型后再处理。
##其中一个想法是将一个属性多种类别，转换成多个属性，每个属性变为0-1类
##使用get_dummies进行one-hot编码
dummies_Cabin=pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked=pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex=pd.get_dummies(data_train['Sex'],prefix='Sex')    
dummies_Pclass=pd.get_dummies(data_train['Pclass'],prefix='Pclass')
dummies_Sex_Pclass=pd.get_dummies(data_train['Sex_Pclass'],prefix='Sex_Pclass')
data_train=pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass,dummies_Sex_Pclass],axis=1)
data_train=data_train.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Cabin_No','Embarked_C','Embarked_Q','Pclass_2','Sex_Pclass'],axis=1)

##预处理——归一化处理
##https://sanwen8.cn/p/135RpHk.html
##TypeError: ‘NoneType’ object is not subscriptable
##错误原因可能是该变量类型有问题
import sklearn.preprocessing as preprocessing 
scaler=preprocessing.StandardScaler()
age_scale_param=scaler.fit(data_train['Age'])
data_train['Age_scaled']=scaler.fit_transform(data_train['Age'],age_scale_param)
fare_scale_param = scaler.fit(data_train['Fare'])
data_train['Fare_scaled'] = scaler.fit_transform(data_train['Fare'],fare_scale_param)
##test
data_test=pd.read_csv('test.csv')
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
data_test['Sex_Pclass']=data_test.Sex+'_'+data_test.Pclass.map(str)
#print(data_test['Sex_Pclass'])
#通过类型的均值填充缺省的位置
Mr=find(data_test.Name,'Mr.')#找到含有x的名字
Mr_age=[]
Mr_ages_index=[]
Mrs=find(data_test.Name,'Mrs.')#找到含有x的名字
Mrs_age=[]
Mrs_ages_index=[]
Miss=find(data_test.Name,'Miss.')#找到含有x的名字
Miss_age=[]
Miss_ages_index=[]
Master=find(data_test.Name,'Master.')#找到含有x的名字
Master_age=[]
Master_ages_index=[]
Dr=find(data_test.Name,'Dr.')#找到含有x的名字
Dr_age=[]
Dr_ages_index=[]
Ms=find(data_test.Name,'Ms.')#找到含有x的名字
Ms_age=[]
Ms_ages_index=[]
data_test=loc_Age_nan(data_test,Mr)
data_test=loc_Age_nan(data_test,Mrs)
data_test=loc_Age_nan(data_test,Miss)
data_test=loc_Age_nan(data_test,Master)
data_test.Age[data_test.Name=="O'Donoghue, Ms. Bridget"]=25
##单加child和mother
data_test['child']=data_test.Age[data_test.Age.values < 12]
data_test.loc[data_test.child.notnull(),'child']=1
data_test.loc[data_test.child.isnull(),'child']=0
               
Mrs=find(data_test.Name,'Mrs.')#找到含有x的名字
print(Mrs)
name_mrs=[]

data_test['mother']=0
for k in range(len(list(Mrs))):
    name_mrs=(data_test.Name[data_test.Name==Mrs[k]].index)
    name_mrs=list(name_mrs)
    print(name_mrs)
#IndexError: too many indices for array
#数组索引太多
    data_test.loc[data_test.mother.index==name_mrs,'mother'] = 1
data_test.mother[data_test.Parch<=1]=0
##特征
data_test = set_Cabin_type(data_test)
data_test
##我们通常会先对类目型的特征因子化/one-hot编码,变成数值型后再处理。
##其中一个想法是将一个属性多种类别，转换成多个属性，每个属性变为0-1类
##使用get_dummies进行one-hot编码
dummies_Cabin=pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Embarked=pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex=pd.get_dummies(data_test['Sex'],prefix='Sex')    
dummies_Pclass=pd.get_dummies(data_test['Pclass'],prefix='Pclass')
dummies_Sex_Pclass=pd.get_dummies(data_test['Sex_Pclass'],prefix='Sex_Pclass')
data_test=pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass,dummies_Sex_Pclass],axis=1)
data_test=data_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Cabin_No','Embarked_C','Embarked_Q','Pclass_2','Sex_Pclass'],axis=1)

##预处理——归一化处理
##https://sanwen8.cn/p/135RpHk.html
##TypeError: ‘NoneType’ object is not subscriptable
##错误原因可能是该变量类型有问题
import sklearn.preprocessing as preprocessing 
scaler=preprocessing.StandardScaler()
age_scale_param=scaler.fit(data_test['Age'])
data_test['Age_scaled']=scaler.fit_transform(data_test['Age'],age_scale_param)
fare_scale_param = scaler.fit(data_test['Fare'])
data_test['Fare_scaled'] = scaler.fit_transform(data_test['Fare'],fare_scale_param)
###模型融合(Bagging)
from sklearn.ensemble import BaggingRegressor
train=data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|child|mother|Embarked_.*|Sex_Pclass_.*')
data_np=train.as_matrix()
##unhashable type: 'slice'数据类型不对，应该是矩阵而非DataFrame(序列结构)
y=data_np[:,0]
x=data_np[:,1:]
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
##将clf传到BaggingRegression
#bagging_clf = BaggingRegressor(clf, n_estimators=10, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1)
clf.fit(x,y)
df_test=data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|child|mother|Embarked_.*|Sex_Pclass_.*')
test=df_test.as_matrix()
predictions=clf.predict(test)
result=pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv('predictions.csv')
##画学习曲线
title="Learning Curves (BaggingRegression)"
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
plot_learning_curve(clf,title,x,y,train_sizes=np.linspace(.1, 1.0, 20),cv=cv, n_jobs=1)
plt.show()
###方法：看看现在得到的模型的系数，因为系数和它们最终的判定能力强弱是正相关的
###（在逻辑回归中有效）
print(list(df_test.columns))
print(list(clf.coef_))





