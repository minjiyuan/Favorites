# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch.nn as nn
from ITMO_FS import UnivariateFilter, f_ratio_measure, select_k_best
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch import optim

from Utils import utils
from Utils.loss import PNTripletloss
from net.AM1 import Adaptive_Cross_Modal_PN
from net.PrototypicalNet import PrototypicalNet

utils.set_seed(1)

def singleModality_cv_10(X,y,embedding_num,support_num,query_num,filter_threshold = 0.1,cv_num = 3):
    acc = []
    auc = []
    f1 = []
    for i in range(cv_num):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state= i)
        cur_acc = []
        cur_auc = []
        cur_f1 = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            s_filter_column = np.sum((X_train != 0), axis=0) > X_train.shape[0] * filter_threshold
            X_train = X_train[:, s_filter_column]
            X_test = X_test[:, s_filter_column]


            s_std = StandardScaler()
            s_std.fit(X_train, y_train)
            X_train = s_std.transform(X_train)
            X_test = s_std.transform(X_test)

            pn = PrototypicalNet(X_train.shape[1], 2, embedding_num, support_num, query_num)
            pn_optimer = optim.Adam(pn.parameters(), lr=0.001, weight_decay=0.001)
            pn_criterion = nn.CrossEntropyLoss()
            pn.fit(X_train, y_train, pn_optimer, pn_criterion, EPOCH=100)
            pre_y, prob_y = pn.predict(X_test)


            cur_acc.append(accuracy_score(pre_y, y_test))
            cur_auc.append(roc_auc_score(y_test, prob_y))
            cur_f1.append(f1_score(y_test, pre_y))

        acc.append(np.mean(np.array(cur_acc)))
        auc.append(np.mean(np.array(cur_auc)))
        f1.append(np.mean(np.array(cur_f1)))

    print("PN Net   ACC:" + str(np.mean(np.array(acc))) + "    AUC：" + str(np.mean(np.array(auc))) +
          "    F1-score：" + str(np.mean(np.array(f1))))

    return {  'ACC': np.mean(np.array(acc)),  'AUC': np.mean(np.array(auc)),  'F1': np.mean(np.array(f1))}



def multiModality_cv_10(X1, X2, y, feature_num, embedding_num,
                        support_num,query_num,filter_threshold1 = 0.1,filter_threshold2 = 0.2,
                        cv_num = 3):
    acc = []
    auc = []
    f1 = []
    for i in range(cv_num):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        cur_acc = []
        cur_auc = []
        cur_f1 = []
        for train_index, test_index in kf.split(X1, y):
            X1_train, X1_test = X1[train_index], X1[test_index]
            X2_train, _ = X2[train_index], X2[test_index]
            y_train, y_test = y[train_index], y[test_index]

            s_filter_column = np.sum((X1_train != 0), axis=0) > X1_train.shape[0] * filter_threshold1
            X1_train = X1_train[:, s_filter_column]
            X1_test = X1_test[:, s_filter_column]

            p_filter_column = np.sum((X2_train != 0), axis=0) > X2_train.shape[0] * filter_threshold2
            X2_train = X2_train[:, p_filter_column]

            s_std = StandardScaler()
            s_std.fit(X1_train, y_train)
            X1_train = s_std.transform(X1_train)
            X1_test = s_std.transform(X1_test)

            p_std = StandardScaler()
            p_std.fit(X2_train, y_train)
            X2_train = p_std.transform(X2_train)

            skb = UnivariateFilter(f_ratio_measure, select_k_best(feature_num))
            skb.fit(X2_train, y_train)
            X2_train = skb.transform(X2_train)

            acmp = Adaptive_Cross_Modal_PN(X1_train.shape[1],
                                           X2_train.shape[1], 2,
                                           embedding_num, support_num, query_num)
            optimer = optim.Adam(acmp.parameters(), lr=0.001, weight_decay=0.001)
            criterion = nn.CrossEntropyLoss()
            acmp.fit(X1_train, X2_train, y_train, optimer, criterion, 100)
            pre_y, prob_y = acmp.predict(X1_test)

            cur_acc.append(accuracy_score(pre_y, y_test))
            cur_auc.append(roc_auc_score(y_test, prob_y))
            cur_f1.append(f1_score(y_test, pre_y))

        acc.append(np.mean(np.array(cur_acc)))
        auc.append(np.mean(np.array(cur_auc)))
        f1.append(np.mean(np.array(cur_f1)))

    print("AM1 Net   ACC:" + str(np.mean(np.array(acc))) + "    AUC：" + str(np.mean(np.array(auc))) +
          "    F1-score：" + str(np.mean(np.array(f1))))

    return {'ACC': np.mean(np.array(acc)), 'AUC': np.mean(np.array(auc)), 'F1': np.mean(np.array(f1))}




def singleModality_embedding_search(species_X,y,embedding_nums,support_num,query_num,filter_threshold,cv_num):

    res_df = pd.DataFrame(index= ["ACC","AUC","F1"], columns=embedding_nums)
    for embedding_num in embedding_nums:
        cur_map = singleModality_cv_10(species_X,y,embedding_num,support_num,query_num,filter_threshold,cv_num=cv_num)
        cur_acc = cur_map['ACC']
        cur_auc = cur_map['AUC']
        cur_f1 =  cur_map['F1']

        res_df.loc["ACC"][embedding_num] = cur_acc
        res_df.loc["AUC"][embedding_num] = cur_auc
        res_df.loc["F1"][embedding_num] = cur_f1

    return res_df


def multiModality_feature_embedding_search(X1, X2, y, feature_nums, embedding_nums,
                        support_num,query_num,filter_threshold1 = 0.1,filter_threshold2 = 0.2,
                        cv_num = 3):


    acc_df = pd.DataFrame(index=feature_nums,
                          columns=embedding_nums)
    auc_df = pd.DataFrame(index=feature_nums,
                          columns=embedding_nums)
    f1_df = pd.DataFrame(index=feature_nums,
                         columns=embedding_nums)

    for feature_num in feature_nums:
        for embedding_num in embedding_nums:
            print(str(feature_num) + "  " + str(embedding_num))
            cur_map = multiModality_cv_10(X1, X2, y, feature_num, embedding_num,
                                          support_num,query_num,filter_threshold1,
                                          filter_threshold2,cv_num)
            cur_acc = cur_map['ACC']
            cur_auc = cur_map['AUC']
            cur_f1 = cur_map['F1']

            acc_df.loc[feature_num][embedding_num] = cur_acc
            auc_df.loc[feature_num][embedding_num] = cur_auc
            f1_df.loc[feature_num][embedding_num] = cur_f1

    return acc_df, auc_df, f1_df



def PNTripletloss_cv_10(X,y,embedding_dim,support_num,query_num,l,margin,filter_threshold=0.1,cv_num=3):
    acc = []
    auc = []
    f1 = []
    for i in range(cv_num):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        cur_acc = []
        cur_auc = []
        cur_f1 = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            s_filter_column = np.sum((X_train != 0), axis=0) > X_train.shape[0] * filter_threshold
            X_train = X_train[:, s_filter_column]
            X_test = X_test[:, s_filter_column]


            s_std = StandardScaler()
            s_std.fit(X_train, y_train)
            X_train = s_std.transform(X_train)
            X_test = s_std.transform(X_test)

            pn = PrototypicalNet(X_train.shape[1], 2, embedding_dim, support_num, query_num)
            pn_optimer = optim.Adam(pn.parameters(), lr=0.001, weight_decay=0.001)
            pn_criterion = PNTripletloss(l=l,margin=margin)
            pn.fit(X_train, y_train, pn_optimer, pn_criterion, EPOCH=100)
            pre_y, prob_y = pn.predict(X_test)

            cur_acc.append(accuracy_score(pre_y, y_test))
            cur_auc.append(roc_auc_score(y_test, prob_y))
            cur_f1.append(f1_score(y_test, pre_y))

        acc.append(np.mean(np.array(cur_acc)))
        auc.append(np.mean(np.array(cur_auc)))
        f1.append(np.mean(np.array(cur_f1)))

    return {  'ACC': np.mean(np.array(acc)),  'AUC': np.mean(np.array(auc)),  'F1': np.mean(np.array(f1))}



def PNTripletloss_search(X,y,embedding_dim,support_num,query_num,L,margins,filter_threshold = 0.1,cv_num=3):

    acc_df = pd.DataFrame(index=L,
                          columns=margins)
    auc_df = pd.DataFrame(index=L,
                          columns=margins)
    f1_df = pd.DataFrame(index=L,
                         columns=margins)
    for l in L:
        for margin in margins:
            cur_map = PNTripletloss_cv_10(X,y,embedding_dim,support_num,query_num,l,margin,
                                          filter_threshold=filter_threshold,cv_num=cv_num)
            cur_acc = cur_map['ACC']
            cur_auc = cur_map['AUC']
            cur_f1 =  cur_map['F1']

            acc_df.loc[l][margin] = cur_acc
            auc_df.loc[l][margin] = cur_auc
            f1_df.loc[l][margin] = cur_f1

    return acc_df,auc_df,f1_df


