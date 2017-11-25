#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: leidelong
@license: Apache Licence 
@contact: leidl8907@gmail.com
@site: https://github.com/JoneNash
@software: PyCharm Community Edition
@time: 2017/11/23 下午4:09
"""


import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

submissions_path = "/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/middle_files"
all_files = os.listdir(submissions_path)

outs = []
for f in all_files:
    tmp_df=pd.read_csv(os.path.join(submissions_path, f), index_col=0)
    print "#######"
    print os.path.join(submissions_path, f)
    outs.append(tmp_df)
concat_df = pd.concat(outs, axis=1)

cols = list(map(lambda x: "target_" + str(x), range(len(concat_df.columns))))
concat_df.columns = cols

# 方法1》Apply ranking, normalization and averaging
# concat_df["target"] = (concat_df.rank() / concat_df.shape[0]).mean(axis=1)

#方法2 》均值
# concat_df["target"] = concat_df.mean(axis=1)

#方法3 》log之后求平均，再指数变换
concat_df["target"] = np.exp((np.log(concat_df)).mean(axis=1))

concat_df.drop(cols, axis=1, inplace=True)



# Write the output
import time
localtime = time.asctime( time.localtime(time.time()) )
concat_df.to_csv("/Users/leidelong/competition/Porto_Seguro_Safe_Driver_Prediction/all_kagglemix"+localtime+".csv.gz",index_label="id",compression = 'gzip')

