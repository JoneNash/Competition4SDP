#coding=utf-8

'''
【TODO】
1.调参
2.扩展模型组合方式
'''

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


train = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/train.csv')
test = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/test.csv')



# Preprocessing 
id_test = test['id'].values
target_train = train['target'].values

#特征工程之后
# train = train.drop(['target','id','tag'], axis = 1)
# test = test.drop(['target','id','tag'], axis = 1)

#原始数据
train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

print(train.values.shape, test.values.shape)



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T) #最终测试集

        #StratifiedKFold()按照类别百分比分组
        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        #转化后的训练集特征、测试集特征：base_models中有N个基分类器则有N列特征
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        #选择一种基分类器
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))
            S_train_i = np.zeros((X.shape[0], self.n_splits))

            #
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))

                y_pred = clf.predict_proba(X_holdout)[:,1]


                #每次赋值1/j的训练数据
                S_train[test_idx, i] = y_pred
                #每次赋值全部的测试数据，所以，后续需要求均值
                S_test_i[:, j] = clf.predict_proba(T)[:,1]

            #虽然做了k折，最终使用k折的均值
            S_test[:, i] = S_test_i.mean(axis=1)

        #将其他分类器的结果作为stacker的输入参数
        results = cross_val_score(self.stacker, S_train, y, cv=10, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res


        
# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 10
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['random_state'] = 99
lgb_params['n_jobs'] = -1

# RandomForest params
rf_params = {}
rf_params['n_estimators'] = 10
rf_params['max_depth'] = 6
rf_params['min_samples_split'] = 70
rf_params['min_samples_leaf'] = 30
rf_params['n_jobs'] = -1

# XGBoost params
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.04
xgb_params['n_estimators'] = 10
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9  
xgb_params['min_child_weight'] = 10
xgb_params['n_jobs'] = -1


lgb_model = LGBMClassifier(**lgb_params)

rf_model = RandomForestClassifier(**rf_params)
        
xgb_model = XGBClassifier(**xgb_params)

log_model = LogisticRegression()


##########    lgb+lgb+lgb     
#Stacker score: 0.80597
#LB 0.252
stack = Ensemble(n_splits=10,
        stacker = log_model,
        base_models = (lgb_model, rf_model, xgb_model))


y_pred = stack.fit_predict(train, target_train, test)        


#输出预测结果
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('stacked_1.csv', index=False)



