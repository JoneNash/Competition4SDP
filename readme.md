# Porto Seguro’s Safe Driver Prediction

[Kaggle比赛：Porto Seguro的安全驾驶预测](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) 代码

## 总结

这次比赛排名不出众，比赛过程中存在以下问题：

* 虽然比赛参与的比较早，但是组队比较晚，分工不明确
* 异地队员交流起来不太流畅
* 第一次参赛不懂套路
* 后期没办法腾出大量时间，发力不足

###方法
1.在比赛前期花了很多时间放在feature engineering上，构建了200个左右的特征，使用xgboost单模型测试，效果比只是用原始特征要好，但是最后阶段并没有使用这部分特征，造成精力上的浪费。

2.feature engineering中使用了特征空值填充、特征组合、one-hot编码。

3.多模型用平均blending的方法融合。blending时采用了几种方法：

* 直接avg
* rank之后求均值
* 取log>均值>exp

4.尝试使用hyperopt对lgb自动调参，scoring设置为赛题组提供的评价方法gini系数，并使用交叉验证。在服务器运行时报错，没有在最终的版本中使用。

5.尝试了KNN，非常慢，时间关系并没有用到最终版本中。转而采用80W测试集在59W训练集中随机采样的方式重新构建训练集。



## [kaggle 1st place](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629)

![](images/1st.result.png)
6个模型做融合（1x lightgbm, 5x nn）.

### feature engineering

作者做的比较少
> 1.Basically I removed *calc, added 1-hot to *cat features. Thats all I've done. No missing value replacement or something. This is featureset "f0" in the table. 
> 
> 2.Thanks to the public kernels (wheel of fortune eg.) that suggest to remove *calc features, I'm too blind and probably would not have figured this out by myself
> 

### local validation

cv,模型结果做均值
>1.5-fold CV as usual. Fixed seed. No stratification. 
>
>2.averages of all fold models. Just standard as I would use for any other task. 
>
>3.Somebody wrote about bagging and its improvements, I spend a week in re-training all my models in a 32-bag setup (sampling with replacement). Score only improved a little.

### normalization

归一化的时候用了RankGauss
>1.Input normalization for gradient-based models such as neural nets is critical.For lightgbm/xgb it does not matter.
>

### unsupervised learning

用无监督模型DAE构造特征，这个阶段往往把特征扩展到1K~10K的数量级
>1.The larger the testset, the better
>
>2.A denoising autoencoder tries to reconstruct the noisy version of the features. It tries to find some representation of the data to better reconstruct the clean one. 
>
>3.I found a noise schema called "swap noise". Here I sample from the feature itself with a certain probability "inputSwapNoise" in the table above. 0.15 means 15% of features replaced by values from another row. 
>

### learning with train+test features unsupervised
略

### other unsupervised models

尝试使用GAN，失败了;开了一个脑洞，没有尝试。
>1.I think they have a fundamental problem in generating both numeric and categoric data.At the end they were low 0.28x on CV, too low to contribute to the blend. Havent tried hard enough.
>
>2.Another idea that come late in my mind was a min/max. game like in GAN to generate good noise samples. Its critical to generate good noise for a DAE. I'm thinking of a generator with feature+noiseVec as input, it maximizes the distance to original sample while the autoencoder (input from generator) tried to reconstruct the sample... more maybe in another competition.

###neural nets

> 1.Hidden layers have 'r' = relu activation
> 
> 2.Trained to minimize logloss
> 
> 3.Input dropout often improve generalization when training on DAE features. 
> 

### lightgbm

>I tuned params on CV.

### blending
Nonlinear things failed.
>For me even tuning of linear blending weights failed. So I stick with all w=1.

### what did not work

>upsampling, deeper autoencoders, wider autoencoders, KNNs, KNN on DAE features, nonlinear stacking, some feature engineering (yes, I tried this too), PCA, bagging, factor models (but others had success with it), xgboost (other did well with that) and much much more.
>


##