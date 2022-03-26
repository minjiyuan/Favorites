# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from Utils.CV import singleModality_embedding_search
from Utils import utils
import warnings

warnings.filterwarnings("ignore")
utils.set_seed(1)

warnings.filterwarnings("ignore")
df = pd.read_table("../../data/curatedMetagenomicData/YuJ_2015/counts/YuJ_2015_counts_species.csv",
                   sep=",", index_col=0)
df = df.dropna(axis=1)
new_columns_names = [x.split('.')[0] + "-" + x.split('.')[1] for x in df.columns]
df.columns = new_columns_names

log_df = df.apply(np.log1p,axis=1).T

label_df = pd.read_table("../../data/curatedMetagenomicData/YuJ_2015/YuJ_2015_pData.csv",
                         sep=",", index_col=0)[['study_condition']]
label_df = label_df.dropna()
label_df.loc[label_df["study_condition"] == "control", "study_condition"] = 0
label_df.loc[label_df["study_condition"] == "CRC", "study_condition"] = 1

data_df = log_df.join(label_df).dropna()

data_arr = np.array(data_df)

X = data_arr[:, :-1].astype(np.float)
y = data_arr[:, -1].astype(np.int)

print(X.shape)
print(y.sum())


res_df = singleModality_embedding_search(X,y, embedding_nums= [24,48,64,128,256,512]
                                         ,support_num=20,query_num=16,
                                         filter_threshold=0.1,cv_num=3)

res_df.to_csv("CRC_search.csv",sep="\t",header=True,index=True)

