#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/11/12 下午3:52
"""

from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

#模型
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams


father_path="/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/"
train_url=father_path+"train.csv"
train_raw=pd.read_csv(train_url)

test_url=father_path+"test.csv"
test_raw=pd.read_csv(test_url)


#拆分数据集
from sklearn.cross_validation import train_test_split

#标注正样本和负样本
train_pos=train_raw[train_raw['target']==1]
print (train_pos['target'].value_counts())
train_neg=train_raw[train_raw['target']==0]
print (train_neg['target'].value_counts())
dtypes = train_raw.dtypes
cols = train_raw.columns
ty = dict(zip(cols,dtypes))
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
train = train.astype(ty)
validation_temp=pd.DataFrame(validation_new)
validation_temp.columns=train_raw.columns
validation=validation_temp
validation = validation.astype(ty)
test=test_raw


#更改df结构
test.insert(1,'target',2)

train.insert(1,'tag','train')
validation.insert(1,'tag','validation')
test.insert(1,'tag','test')

alldata=pd.concat([train,validation,test]) #在纵轴上合并
#存储中间结果
train_validation_tag=pd.concat([train,validation])
train_validation_tag.to_csv(father_path+"train_validation_tag.csv",index=False)


#创建元数据
data = []
for f in train.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    elif f == 'tag':
        role = 'tag'
    else:
        role = 'input'

    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f in {'id', 'tag'}:
        level = 'nominal'
    elif train[f].dtype == float:
        level = 'interval'
    elif train[f].dtype == int:
        level = 'ordinal'

    # Initialize keep to True for all variables except for id
    keep = True
    if f in {'id', 'tag'}:
        keep = False

    # Defining the data type
    dtype = train[f].dtype

    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)

pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()

target='target'
IDcol = 'id'
tag='tag'
predictors = [x for x in alldata.columns if x not in [target, IDcol,tag]]

print ('空值占比：')
line_count=alldata['target'].count()
for i in predictors :
    nan_cnt=  line_count - alldata[i][alldata[i] != -1].count()
    if nan_cnt!=0:
        print (i,": {:.5f}% ".format(nan_cnt/(line_count*1.0)*100 ))