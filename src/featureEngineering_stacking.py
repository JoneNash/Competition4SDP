#coding=utf-8

'''
效果并不好，LB 0.240。

肯定是某个地方出错了...
'''

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

# Regularized Greedy Forest
from rgf.sklearn import RGFClassifier     # https://github.com/fukatani/rgf_python


train = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/train_extends.csv')
test = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/test_extends.csv')



# Preprocessing 
id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target','id','tag'], axis = 1)
test = test.drop(['target','id','tag'], axis = 1)

print(train.values.shape, test.values.shape)



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

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

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res


        
# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['random_state'] = 99


lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['random_state'] = 99


lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['random_state'] = 99


# RandomForest params
rf_params = {}
rf_params['n_estimators'] = 200
rf_params['max_depth'] = 6
rf_params['min_samples_split'] = 70
rf_params['min_samples_leaf'] = 30


# ExtraTrees params
et_params = {}
et_params['n_estimators'] = 155
et_params['max_features'] = 0.3
et_params['max_depth'] = 6
et_params['min_samples_split'] = 40
et_params['min_samples_leaf'] = 18


# XGBoost params
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.04
xgb_params['n_estimators'] = 490
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9  
xgb_params['min_child_weight'] = 10



# XGBoost params,with scale_pos_weight
xgb_weight_params = {}
xgb_weight_params['objective'] = 'binary:logistic'
xgb_weight_params['learning_rate'] = 0.04
xgb_weight_params['n_estimators'] = 490
xgb_weight_params['max_depth'] = 4
xgb_weight_params['subsample'] = 0.9
xgb_weight_params['colsample_bytree'] = 0.9  
xgb_weight_params['min_child_weight'] = 4
xgb_weight_params['scale_pos_weight'] = 26


# CatBoost params
cat_params = {}
cat_params['iterations'] = 900
cat_params['depth'] = 8
cat_params['rsm'] = 0.95
cat_params['learning_rate'] = 0.03
cat_params['l2_leaf_reg'] = 3.5  
cat_params['border_count'] = 8
cat_params['gradient_iterations'] = 4


# Regularized Greedy Forest params
rgf_params = {}
rgf_params['max_leaf'] = 2000
rgf_params['learning_rate'] = 0.5
rgf_params['algorithm'] = "RGF_Sib"
rgf_params['test_interval'] = 100
rgf_params['min_samples_leaf'] = 3 
rgf_params['reg_depth'] = 1.0
rgf_params['l2'] = 0.5  
rgf_params['sl2'] = 0.005



lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

rf_model = RandomForestClassifier(**rf_params)

et_model = ExtraTreesClassifier(**et_params)
        
xgb_model = XGBClassifier(**xgb_params)

xgb_weight_model = XGBClassifier(**xgb_weight_params)

cat_model = CatBoostClassifier(**cat_params)

rgf_model = RGFClassifier(**rgf_params) 

gb_model = GradientBoostingClassifier(max_depth=5)

ada_model = AdaBoostClassifier()

log_model = LogisticRegression()


##########    lgb+lgb+lgb     
stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2, lgb_model3)) 


y_pred = stack.fit_predict(train, target_train, test)        



sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('stacked_1.csv', index=False)



##########    lgb+xgb
stack = Ensemble(n_splits=5,
        stacker = log_model,
        base_models = (lgb_model, xgb_model))

y_pred = stack.fit_predict(train, target_train, test)        



sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('stacked_2.csv', index=False)


########## lgb_model + xgb_weight_model
stack = Ensemble(n_splits=5,
        stacker = log_model,
        base_models = (lgb_model, xgb_weight_model))

y_pred = stack.fit_predict(train, target_train, test)        

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('stacked_3.csv', index=False)



#大量模型

stack = Ensemble(n_splits=5,
        stacker = log_model,
        base_models = (lgb_model,lgb_model2,lgb_model3,rf_model,et_model,xgb_model,xgb_weight_model,cat_model,rgf_model,gb_model,ada_model))

y_pred = stack.fit_predict(train, target_train, test)        

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('stacked_4.csv', index=False)




