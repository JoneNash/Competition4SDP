#coding=utf-8

import pandas as pd
import numpy as np

#评估方法
def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses
 
     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)

def gini_normalized(a, p):
        return gini(a, p) / gini(a, a)
    


#模型训练
def xgboostModelFit(alg, dtrain, dtest, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #模型训练
    alg.fit(dtrain[predictors], dtrain['target'],eval_metric='auc')
        
    #模型预测-训练集
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #输出模型效果-训练集:
    print "\nModel Report- train data"
    print "Accuracy（Train） : %.4g" % metrics.accuracy_score(dtrain['target'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['target'], dtrain_predprob)
    print "GINI Score (Train): %f" % gini_normalized(dtrain['target'], dtrain_predprob)
    
    #模型预测-测试集
    dtest['prediction'] = alg.predict(dtest[predictors])
    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:,1]
    #输出模型效果-测试集
    print "\nModel Report- test data"
    print "Accuracy（Test） : %.4g" % metrics.accuracy_score(dtest['target'].values,dtest['prediction'])
    print 'AUC Score (Test): %f' % metrics.roc_auc_score(dtest['target'], dtest['predprob'])
    print "GINI Score (Test): %f" % gini_normalized(dtest['target'], dtest['predprob'])
    
    return alg





train1 = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/train.csv')
test1 = pd.read_csv("/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/test.csv")

train2 = pd.read_csv('/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/train_extends.csv')
test2 = pd.read_csv("/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/input/test_extends.csv")

target='target'
IDcol = 'id'
tag='tag'


#模型
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV




predictors1 = [x for x in train1.columns if x not in [target, IDcol,tag]]

xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=20,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=26.4, #解决样本不均衡的问题
        seed=27)

alg_before=xgboostModelFit(xgb1, train1, train1, predictors1)



predictors2 = [x for x in train2.columns if x not in [target, IDcol,tag]]

xgb2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=20,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=26.4, #解决样本不均衡的问题
        seed=27)

alg_after=xgboostModelFit(xgb2, train2, train2, predictors2)
