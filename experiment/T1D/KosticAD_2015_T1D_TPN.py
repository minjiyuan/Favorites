# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from Utils.CV import PNTripletloss_search
from Utils import utils
import warnings

warnings.filterwarnings("ignore")
utils.set_seed(1)

df = pd.read_table("../../data/curatedMetagenomicData/KosticAD_2015/counts/KosticAD_2015_counts_species.csv",
                   sep=",", index_col=0)

df = df.dropna(axis=1)

log_df = df.apply(np.log1p,axis=1).T


label_df = pd.read_table("../../data/curatedMetagenomicData/KosticAD_2015/KosticAD_2015_pData.csv",
                         sep=",", index_col=0)[["subjectID"]]
label_df = label_df.dropna()
label_df.loc[label_df["subjectID"] == "control", "subjectID"] = 0
label_df.loc[label_df["subjectID"] == "T1D", "subjectID"] = 1

data_df = log_df.join(label_df).dropna()

data_arr = np.array(data_df)

X = data_arr[:, :-1].astype(np.float)
y = data_arr[:, -1].astype(np.int)

print(X.shape)
print(y.sum())



acc_df,auc_df,f1_df = PNTripletloss_search(X,y,embedding_dim=24,
                                           support_num=30,query_num=20,
                                           L=[0.05,0.1,0.2,0.3,0.4,0.5],
                                          margins=[1,2,3,4,5,6,7,8,9,10]
                                           ,filter_threshold=0.1, cv_num=3)

acc_df.to_csv("MixtureLoss/T1D_MixtureLoss_ACC.csv",sep="\t",header=True,index=True)
auc_df.to_csv("MixtureLoss/T1D_MixtureLoss_AUC.csv",sep="\t",header=True,index=True)
f1_df.to_csv("MixtureLoss/T1D_MixtureLoss_f1.csv",sep="\t",header=True,index=True)
