#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

#模型
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


train_url="/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/train.csv"
train_raw=pd.read_csv(train_url)

test_url="/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/test.csv"
test_raw=pd.read_csv(test_url)

print "train shape:",train_raw.shape
print "test shape:",test_raw.shape
print "train columns:",train_raw.columns
print "test columns:",test_raw.columns

#拆分数据集
from sklearn.cross_validation import train_test_split  

#标注正样本和负样本
train_pos=train_raw[train_raw['target']==1]
print train_pos['target'].value_counts()
train_neg=train_raw[train_raw['target']==0]
print train_neg['target'].value_counts()

#df转narray
train_pos=np.array(train_pos)
train_neg=np.array(train_neg)
#数据集拆分
train_pos_new,validation_pos_new = train_test_split(train_pos,  
                                                   test_size = 0.2,  
                                                   random_state = 0)  
train_neg_new,validation_neg_new = train_test_split(train_neg,  
                                                   test_size = 0.2,  
                                                   random_state = 0)  
#数据集拼接
train_new=np.concatenate([train_pos_new,train_neg_new],axis=0) #在纵轴上合并

validation_new=np.concatenate([validation_pos_new,validation_neg_new],axis=0) #在纵轴上合并


#修改train、validation、test
train_temp=pd.DataFrame(train_new)
train_temp.columns=train_raw.columns
train=train_temp

validation_temp=pd.DataFrame(validation_new)
validation_temp.columns=train_raw.columns
validation=validation_temp

test=test_raw



