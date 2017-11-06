#特征工程-单模型测试效果
特征工程之后，有效特征172个。

删除的特征：

* ps_car_03_cat：删除
* ps_car_05_cat：删除
* ps_reg_03：均值填充
* ps_car_12：均值填充
* ps_car_14：均值填充
* ps_car_11：（int型）最高频率填充
* ps_car_11_cat：类别过多，转为对应target均值，原始特征删除

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