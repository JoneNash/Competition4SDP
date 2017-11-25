#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/11/25 上午11:48
"""


#coding=utf-8
# from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regularized Greedy Forest
from rgf.sklearn import RGFClassifier     # https://github.com/fukatani/rgf_python
from hyperopt import fmin, tpe, hp, rand

train = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/train.csv')
test = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/test.csv')



# Preprocessing
id_test = test['id'].values
target_train = train['target'].values

#原始数据
train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)


from sklearn.cross_validation import train_test_split
#x为数据集的feature熟悉，y为label.
x_train, x_test, y_train, y_test = train_test_split(train, target_train, test_size = 0.2 ,random_state=1)



print(train.values.shape, test.values.shape)



# 计数器，每一次参数组合的枚举都会使它加1
count = 0


# 评估方法
def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def function(args):
    print(args)

    # **可以把dict转换为关键字参数，可以大大简化复杂的函数调用
    clf =  LGBMClassifier(**args)

    lgb_params = {}
    # lgb_params['n_estimators'] = 10
    # lgb_params['max_bin'] = 10
    # lgb_params['subsample'] = 0.8
    # lgb_params['subsample_freq'] = 10
    # lgb_params['colsample_bytree'] = 0.8
    # lgb_params['min_child_samples'] = 500
    # lgb_params['random_state'] = 99
    lgb_params['n_jobs'] = -1

    clf.set_params(**lgb_params)

    # 训练模型
    clf.fit(x_train,y_train)

    # 预测测试集
    prediction_proba = clf.predict_proba(x_test)[:,1]

    global count
    count = count + 1

    # score = accuracy_score(y_test, prediction)
    score = gini_normalized(y_test,prediction_proba)
    print(" %s , test accuracy : " % str(count),score)

    # # 由于hyperopt仅提供fmin接口，因此如果要求最大值，则需要取相反数
    return -score



# 计数器，每一次参数组合的枚举都会使它加1
count = 0

parameter_space_lgb ={
    'n_estimators':hp.choice('n_estimators',range(500,1501,1)),
    'max_features':hp.choice('n_estimators',range(7,14,1)),
    'n_jobs':-1,

}

best = fmin(function, parameter_space_lgb, algo=tpe.suggest, max_evals=10)

clf = LGBMClassifier(**best)
print(clf)


#验证最优参数效果
# 训练模型
clf.fit(x_train,y_train)
prediction_proba = clf.predict_proba(x_test)[:,1]
score = gini_normalized(y_test,prediction_proba)
print(" best params : %s , test accuracy : %s" % (best,score))


#全体数据参与训练，并输出测试集结果
clf.fit(train, target_train)
prediction_proba = clf.predict_proba(test)[:,1]
#输出预测结果
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = prediction_proba
sub.to_csv('stacked_1.csv', index=False)
