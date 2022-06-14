# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch import optim
from net.PrototypicalNet import PrototypicalNet

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

def miv(model, X):
    model.eval()
    miv = torch.ones(X.shape[1])
    for i in range(X.shape[1]):

        cur_X_1 = X.copy()
        cur_X_2 = X.copy()

        cur_X_1[:, i] = cur_X_1[:, i] + cur_X_1[:, i] * 0.1
        cur_X_2[:, i] = cur_X_2[:, i] - cur_X_2[:, i] * 0.1

        cur_X_1 = np.log1p(cur_X_1)
        cur_X_2 = np.log1p(cur_X_2)
        std = StandardScaler()
        cur_X_1 = std.fit_transform(cur_X_1)
        cur_X_2 = std.fit_transform(cur_X_2)
        cur_X_1 = torch.tensor(cur_X_1, dtype=torch.float)
        cur_X_2 = torch.tensor(cur_X_2, dtype=torch.float)

        cur_diff = torch.sum(model.embedding(cur_X_1) - model.embedding(cur_X_2), dim=1)

        miv[i] = torch.mean(torch.abs(cur_diff), dim=0)

    s = miv / torch.sum(miv)
    rank = torch.argsort(torch.abs(miv), dim=0, descending=True)
    return rank, s




if __name__ == '__main__':
    df = pd.read_table("../data/curatedMetagenomicData/YuJ_2015/counts/YuJ_2015_counts_species.csv",
                       sep=",", index_col=0)
    df = df.dropna(axis=1)
    new_columns_names = [x.split('.')[0] + "-" + x.split('.')[1] for x in df.columns]
    df.columns = new_columns_names
    flag_df = (df != 0).sum(axis=1) > df.shape[1] * 0.2
    filted_df = df[flag_df]

    ##### log转换
    log_df = filted_df.T

    label_df = pd.read_table("../data/curatedMetagenomicData/YuJ_2015/YuJ_2015_pData.csv",
                             sep=",", index_col=0)[['study_condition']]

    label_df = label_df.dropna()
    label_df.loc[label_df["study_condition"] == "control", "study_condition"] = 0
    label_df.loc[label_df["study_condition"] == "CRC", "study_condition"] = 1

    data_df = log_df.join(label_df)
    data_df = data_df.dropna()
    data_arr = np.array(data_df)
    X = data_arr[:, :-1].astype(float)
    Y = data_arr[:, -1].astype(float)

    print(X.shape)
    print(Y.sum())

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=1234)

    X_train = np.log1p(X_train)
    X_test = np.log1p(X_test)
    std = StandardScaler()
    std.fit(X_train)
    X_train = std.transform(X_train)
    X_test = std.transform(X_test)

    model = PrototypicalNet(in_feature=X_train.shape[1], num_class=2,embedding_dim=24,
                            support_num=20,query_num=10)
    optimer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    model.fit(X_train,Y_train,optimer,criterion,100)
    pre_y ,_ = model.predict(X_test)

    X_train, X_test, Y_train, _ = train_test_split(X, Y, test_size=0.3,
                                                        random_state=1234)

    rank,s = miv(model,X_train)
    print(rank)
    print(s)
    import matplotlib.pyplot as plt
    import seaborn as sns

    f_importance = pd.DataFrame({"feature": filted_df.index,"importance": s.detach().numpy()})
    f_importance = f_importance.sort_values(by="importance", ascending=False)
    print(f_importance)
    f_importance = f_importance[:10]
    sns.barplot(x="importance", y="feature", data=f_importance,
                order=f_importance["feature"],orient="h",palette=sns.color_palette("tab10", 10))
    plt.show()

