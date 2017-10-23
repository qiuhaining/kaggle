# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:35:21 2017

@author: qiu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_origin=pd.read_csv('kaggle_bike_competition_train.csv',header=0)
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
#拆分时间
df_origin['month']=pd.DatetimeIndex(df_origin.datetime).month
df_origin['day']=pd.DatetimeIndex(df_origin.datetime).dayofweek
df_origin['hour']=pd.DatetimeIndex(df_origin.datetime).hour
ndf_origin=df_origin
read_data(ndf_origin)
##丢掉暂时不用的数据
df_train=ndf_origin.drop(['datetime','registered'],axis=1)
####拆分数据为输入输出
df_train_target=df_train['count'].values
df_train_data=df_train.drop(['count'],axis = 1).values
print ('df_train_data shape is ', df_train_data.shape)
print ('df_train_target shape is ', df_train_target.shape)
#analysis_data
####属性X与Y之间的关系(图)
##连续量
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

df_train.groupby('windspeed').mean().plot(y='count',marker='o')
plt.show()

df_train.groupby('temp').mean().plot(y='count',marker='o')
plt.show()

df_train.groupby('atemp').mean().plot(y='count',marker='o')
plt.show()

df_train.groupby('humidity').mean().plot(y='count',marker='o')
plt.show()

df_train.groupby('casual').mean().plot(y='count',marker='o')
plt.show()
##散点图
# scatter一下各个维度
fig, axs = plt.subplots(4, 3, sharey=True)
df_train.plot(kind='scatter', x='temp', y='count', ax=axs[0, 0], figsize=(16, 8), color='magenta')
df_train.plot(kind='scatter', x='atemp', y='count', ax=axs[0, 1], color='cyan')
df_train.plot(kind='scatter', x='humidity', y='count', ax=axs[0, 2], color='red')
df_train.plot(kind='scatter', x='windspeed', y='count', ax=axs[1, 0], color='yellow')
df_train.plot(kind='scatter', x='month', y='count', ax=axs[1, 1], color='blue')
df_train.plot(kind='scatter', x='hour', y='count', ax=axs[1, 2], color='green')
df_train.plot(kind='scatter', x='season', y='count', ax=axs[2, 0], color='red')
df_train.plot(kind='scatter', x='holiday', y='count', ax=axs[2, 1], color='yellow')
df_train.plot(kind='scatter', x='workingday', y='count', ax=axs[2, 2], color='blue')
df_train.plot(kind='scatter', x='weather', y='count', ax=axs[3, 1], color='green')
#离散量


fig=plt.figure()
fig.set(alpha=0.2)
#plt.subplot2grid((2,2),(0,0))


#df_train.season.values_count
count_season_1=df_train.holiday[df_train.season == 1].value_counts()
count_season_2=df_train.holiday[df_train.season == 2].value_counts()
count_season_3=df_train.holiday[df_train.season == 3].value_counts()
count_season_4=df_train.holiday[df_origin.season == 4].value_counts()
df=pd.DataFrame({'第一季':count_season_1,'第二季':count_season_2,'第三季':count_season_3,'第四季':count_season_4,})
df.plot(kind='bar',stacked=True) 
plt.title("租车数量")
plt.ylabel("租车数量")
plt.show()
corr = df_train[['temp','weather','windspeed','day', 'month', 'hour','count']].corr()
corr
# 用颜色深浅来表示相关度矩阵
plt.figure()
plt.matshow(corr)
plt.colorbar()
plt.show()
####机器学习，学习模型
#输入库
from sklearn import svm,linear_model
#from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit,train_test_split
from sklearn.metrics import explained_variance_score
#切分数据(交叉验证策略)
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
##训练模型
#岭回归
print ("岭回归")#L2正则化的最小二乘法
##广义线性模型
##http://scikit-learn.org/stable/modules/linear_model.html
for train, test in cv.split(df_train_data):    
    svc = linear_model.Ridge().fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test]))) 
print ("支持向量回归/SVR(kernel='rbf',C=10,gamma=.001)")
for train, test in cv.split(df_train_data): 
    svc=svm.SVR(kernel='rbf',C=10,gamma=.001).fit(df_train_data[train], df_train_target[train])
    print("train score :{0:.3f},test score:{1:.3f}\n".format(
          svc.score(df_train_data[train], df_train_target[train]),svc.score(df_train_data[test], df_train_target[test])))
print ("随机森林回归/Random Forest(n_estimators = 100)"  )   
for train, test in cv.split(df_train_data):    
    svc = RandomForestRegressor(n_estimators = 100).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))
##交叉验证选择超参数
X = df_train_data
y = df_train_target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
#C_range=np.logspace(-2, 10, 13)
#gamma_range=np.logspace(-9, 3, 13)
#param_grid=dict(gamma=gamma_range, C=C_range)
#S_svc=svm.SVR(kernel='rbf')
#clf = GridSearchCV(S_svc, param_grid=param_grid,cv=cv)
#clf.fit(X_train, y_train)
#print("The best parameters are %s with a score of %0.2f"
#      % (clf.best_params_, clf.best_score_))#找到最佳超参数

##随机森林选取最佳超参数      
tuned_parameters = [{'n_estimators':[10,100,500]}]      
scores = ['r2']#模型评价工具，r2是指线性相关系数
#http://scikit-learn.org/stable/modules/model_evaluation.html
for score in scores:    
    print (score)    
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    print("得分分别是:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print ("")
### 看看学习情况如何(画学习曲线)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)#*用来传递任意个无名字参数，这些参数会一个元组的形式访问。
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores=learning_curve(estimator,X,y,cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot( train_sizes, train_scores_mean,'o-',linewidth=0.5,color='r',label="Training score")
    plt.plot( train_sizes, test_scores_mean,'o-',color='g',linewidth=0.5,label="Cross-validation score")
    plt.legend(loc="best")
    return plt
   
title = "Learning Curves (Random Forest, n_estimators = 100)"
cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
estimator = RandomForestRegressor(n_estimators = 100)
print("画学习曲线")
plot_learning_curve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=1)
plt.show()

#发现过拟合，解决过拟合问题
# 尝试一下缓解过拟合，当然，未必成功
print ("随机森林回归/Random Forest(n_estimators=200, max_features=0.6, max_depth=15)")
for train, test in cv.split(df_train_data):
    svc = RandomForestRegressor(n_estimators = 200, max_features=0.6, max_depth=15).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))




