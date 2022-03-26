# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from Utils.CV import multiModality_feature_embedding_search
from Utils import utils
import warnings
utils.set_seed(1)


warnings.filterwarnings('ignore')

marker_df = pd.read_csv("../../data/curatedMetagenomicData/YuJ_2015/YuJ_2015.marker_presence.csv",
                             sep=',', index_col=0).T

print(marker_df.head())
print(marker_df.shape)

new_index_names = [x.split('.')[0] + "-" + x.split('.')[1] for x in marker_df.index]
marker_df.index = new_index_names


label_df = pd.read_table("../../data/curatedMetagenomicData/YuJ_2015/YuJ_2015_pData.csv",
                         sep=",", index_col=0)[['study_condition']]
label_df = label_df.dropna()
label_df.loc[label_df["study_condition"] == "control", "study_condition"] = 0
label_df.loc[label_df["study_condition"] == "CRC", "study_condition"] = 1

data_df1 = marker_df.iloc[:, 2:].join(label_df).dropna()

data_arr1 = np.array(data_df1)

marker_X = data_arr1[:, 2:-1].astype(np.float)

species_df = pd.read_table("../../data/curatedMetagenomicData/YuJ_2015/counts/YuJ_2015_counts_species.csv",
                   sep=",", index_col=0).dropna(axis=1)

species_df = species_df.dropna(axis=1)
new_columns_names = [x.split('.')[0] + "-" + x.split('.')[1] for x in species_df.columns]
species_df.columns = new_columns_names

log_df = species_df.apply(np.log1p, axis=1).T
data_df2 = log_df.join(label_df).dropna()
data_arr2 = np.array(data_df2)

species_X = data_arr2[:, :-1].astype(np.float)
y = data_arr2[:, -1].astype(np.int)


acc_df, auc_df, f1_df = multiModality_feature_embedding_search(species_X, marker_X, y,
                                                 feature_nums=[12, 24, 48, 64, 128, 256, 512, 1024],
                                                 embedding_nums=[24, 48, 64, 128, 256, 512],
                                                               support_num=20,query_num=16,
                                                               filter_threshold1=0.1,
                                                               filter_threshold2=0.2,cv_num=3)


acc_df.to_csv("marker/CRC_search_ACC_marker.csv",sep="\t",header=True,index=True)
auc_df.to_csv("marker/CRC_search_AUC_marker.csv",sep="\t",header=True,index=True)
f1_df.to_csv("marker/CRC_search_f1_marker.csv",sep="\t",header=True,index=True)
