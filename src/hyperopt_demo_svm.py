#coding:utf-8
from hyperopt import fmin, tpe, hp, rand
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import datasets
"""
http://blog.csdn.net/gg_18826075157/article/details/78068086
"""
# SVM的三个超参数：C为惩罚因子，kernel为核函数类型，gamma为核函数的额外参数（对于不同类型的核函数有不同的含义）
# 有别于传统的网格搜索（GridSearch），这里只需要给出最优参数的概率分布即可，而不需要按照步长把具体的值给一个个枚举出来
parameter_space_svc ={
    # loguniform表示该参数取对数后符合均匀分布
    'C':hp.loguniform("C", np.log(1), np.log(100)),
    'kernel':hp.choice('kernel',['rbf','poly']),
    'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
}

# 鸢尾花卉数据集，是一类多重变量分析的数据集
# 通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类
iris = datasets.load_digits()

#--------------------划分训练集和测试集--------------------
train_data = iris.data[0:1300]
train_target = iris.target[0:1300]
test_data = iris.data[1300:-1]
test_target = iris.target[1300:-1]
#-----------------------------------------------------------

# 计数器，每一次参数组合的枚举都会使它加1
count = 0

def function(args):
    print(args)

    # **可以把dict转换为关键字参数，可以大大简化复杂的函数调用
    clf = svm.SVC(**args)

    # 训练模型
    clf.fit(train_data,train_target)

    # 预测测试集
    prediction = clf.predict(test_data)

    global count
    count = count + 1
    score = accuracy_score(test_target,prediction)
    print(" %s , test accuracy : " % str(count),score)

    # 由于hyperopt仅提供fmin接口，因此如果要求最大值，则需要取相反数
    return -score

# algo指定搜索算法，目前支持以下算法：
# ①随机搜索(hyperopt.rand.suggest)
# ②模拟退火(hyperopt.anneal.suggest)
# ③TPE算法（hyperopt.tpe.suggest，算法全称为Tree-structured Parzen Estimator Approach）
# max_evals指定枚举次数上限，即使第max_evals次枚举仍未能确定全局最优解，也要结束搜索，返回目前搜索到的最优解
best = fmin(function, parameter_space_svc, algo=tpe.suggest, max_evals=100)

# best["kernel"]返回的是数组下标，因此需要把它还原回来
kernel_list = ['rbf','poly']
best["kernel"] = kernel_list[best["kernel"]]

print("best params: ",best)

clf = svm.SVC(**best)
print(clf)

# 训练模型
clf.fit(train_data,train_target)

# 预测测试集
prediction = clf.predict(test_data)
score = accuracy_score(test_target,prediction)
print(" best params : %s, test accuracy : %s" % (str(count),score))