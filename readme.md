#特征工程-单模型测试效果
特征工程之后，有效特征172个。

特征工程之前的效果：

* Accuracy（Train） : 0.6274
* AUC Score (Train): 0.644518
* GINI Score (Train): 0.289036
* Accuracy（Test） : 0.6281
* AUC Score (Test): 0.644204
* GINI Score (Test): 0.288419


特征工程之后的效果：

* Accuracy（Train） : 0.6251
* AUC Score (Train): 0.648900
* GINI Score (Train): 0.297813
* Accuracy（Test） : 0.6234
* AUC Score (Test): 0.630744
* GINI Score (Test): 0.261501

特征工程前后模型参数相同，都没有经过调参。
在leaderboard上，特征工程之前0.260，特征工程之后0.250。

#特征工程-多模型测试效果
在leaderboard的得分：

* stack_1:0.278
* stack_2:0.278
* stack_3:0.275
* stack_4:0.275

由于为做任何调参处理，效果都低于原始版本的0.284。