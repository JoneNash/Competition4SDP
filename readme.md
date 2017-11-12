# 特征工程-单模型测试效果

1.有关哑变量或one-hot编码

一个错误的想法：xgboost属于树模型，不需要做哑变量和one-hot编码

使用哑变量之后，xgboost的效果变得更好

将取值个数较少的特征转为哑变量，将取值过多的特征转为target的均值，60个特征变为114个特征，效果变化：
之前——

* Accuracy（Train） : 0.6389
* AUC Score (Train): 0.647110
* GINI Score (Train): 0.294231
* Accuracy（Test） : 0.6381
* AUC Score (Test): 0.631639
* GINI Score (Test): 0.263289

之后——

* Accuracy（Train） : 0.6381
* AUC Score (Train): 0.649175
* GINI Score (Train): 0.298352
* Accuracy（Test） : 0.6365
* AUC Score (Test): 0.632581
* GINI Score (Test): 0.265163

2.关于缺失值填充

使用均值或最高频填充之后，整体效果均有所下降，可以认为填充不合理。

* Accuracy（Train） : 0.6332
* AUC Score (Train): 0.648476
* GINI Score (Train): 0.296953
* Accuracy（Test） : 0.6311
* AUC Score (Test): 0.630581
* GINI Score (Test): 0.261163

接下来的操作中，不对缺失值做填充。

3.有关多项式特征组合


* Accuracy（Train） : 0.6348
* AUC Score (Train): 0.652642
* GINI Score (Train): 0.305284
* Accuracy（Test） : 0.6326
* AUC Score (Test): 0.632502
* GINI Score (Test): 0.265006

与1相比，AUC得分有所提升，GINI score突破0.3，基本可以认为特征组合有一定效果。


特征重要性重复比较多，初步判断为a**2 等组合方式与a重复引起，后续考虑特征选择，将这部分特征剔除。

单模型测试，特征工程前后LB均为0.275.

# 特征工程-多模型测试效果

在leaderboard的得分：

* stack_1:
* stack_2:
* stack_3:
* stack_4:

由于为做任何调参处理，效果都低于原始版本的...。