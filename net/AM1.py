# -*- coding: utf-8 -*-
import warnings
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import  StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import optim
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)


"""

Adaptive_Cross_Modal_PN is the implementation of AMPN in the paper

"""

class Adaptive_Cross_Modal_PN(nn.Module):
    def __init__(self, species_in_feature, pathway_in_feature,num_class, embedding_dim,
                 support_num, query_num, distance='euclidean', mixup_data=False):
        super(Adaptive_Cross_Modal_PN, self).__init__()

        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.support_num = support_num
        self.query_num = query_num
        self.distance = distance
        self.prototypical = None
        self.prototypes = []
        self.mixup_data = mixup_data

        self.species_feature_extraction = nn.Sequential(
            nn.Linear(in_features=species_in_feature, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=embedding_dim),
        )

        self.pathway_feature_extraction = nn.Sequential(
            nn.Linear(in_features=pathway_in_feature, out_features=pathway_in_feature * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=pathway_in_feature * 2, out_features=embedding_dim)
        )

        self.fusion_factor = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features= embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim),
        )

    def forward(self, species_support_input,pathway_support_input,
                species_query_input,pathway_query_input):

        species_support_embedding = self.species_feature_extraction(species_support_input)
        species_query_embedding = self.species_feature_extraction(species_query_input)
        pathway_support_embedding = self.pathway_feature_extraction(pathway_support_input)
        # pathway_query_embedding = self.pathway_feature_extraction(pathway_query_input)
        support_size = species_support_embedding.shape[0]
        every_class_num = support_size // self.num_class

        negtive_pathway_prototype = torch.mean(pathway_support_embedding[0:every_class_num, :], dim=0)
        postive_pathway_prototype= torch.mean(pathway_support_embedding[every_class_num:, :], dim=0)

        negtive_factor = self.fusion_factor(negtive_pathway_prototype)
        postive_factor = self.fusion_factor(postive_pathway_prototype)

        negtive_factor = 1 / (1 + torch.exp(-1 * negtive_factor))
        postive_factor = 1 / (1 + torch.exp(-1 * postive_factor))

        class_meta_dict = {}

        negtive_species_prototype= torch.mean(species_support_embedding[0 : every_class_num, :], dim=0)


        class_meta_dict[0] = postive_factor* negtive_species_prototype + \
                        (1 - postive_factor) * negtive_pathway_prototype

        postive_species_prototype = torch.mean(species_support_embedding[every_class_num:, :], dim=0)

        class_meta_dict[1] = negtive_factor * postive_species_prototype + \
                             (1 - negtive_factor) * postive_pathway_prototype


        class_meta_information = torch.zeros(size=[len(class_meta_dict), species_support_embedding.shape[1]])

        for key, item in class_meta_dict.items():
            class_meta_information[key, :] = class_meta_dict[key]

        self.prototypical = class_meta_information

        self.prototypes.append(class_meta_information.detach().numpy())

        N_query = species_query_embedding.shape[0]
        result = torch.zeros(size=[N_query, self.num_class])


        for i in range(0, N_query):
            temp_value = species_query_embedding[i].repeat(self.num_class, 1)
            dist_value = 0
            if self.distance == 'euclidean':
                dist_value = F.pairwise_distance(self.prototypical, temp_value, p=2)
            elif self.distance == 'cosine':
                dist_value = torch.cosine_similarity(self.prototypical, temp_value, dim=1)
                dist_value = 1 - dist_value
            result[i] = -1 * dist_value

        return result

    def randomGenerate(self, species_X,pathway_X, Y):

        postive_index = np.where(Y == 1)[0]
        negtive_index = np.where(Y == 0)[0]

        pos_support_index = np.random.choice(postive_index, self.support_num // 2, replace=False)

        neg_support_index = np.random.choice(negtive_index, self.support_num // 2, replace=False)

        support_index = np.concatenate((neg_support_index, pos_support_index), axis=0)
        species_support_input = species_X[support_index, :]
        pathway_support_input = pathway_X[support_index, :]
        support_label = Y[support_index]

        pos_query_index = np.random.choice([index for index in postive_index if index not in pos_support_index],
                                           self.query_num // 2, replace=False)
        neg_query_index = np.random.choice([index for index in negtive_index if index not in neg_support_index],
                                           self.query_num // 2, replace=False)
        query_index = np.concatenate((neg_query_index, pos_query_index), axis=0)
        species_query_input = species_X[query_index,:]
        pathway_query_input = pathway_X[query_index,:]
        query_label = Y[query_index]

        species_support_input = torch.tensor(species_support_input, dtype=torch.float)
        pathway_support_input = torch.tensor(pathway_support_input, dtype=torch.float)
        species_query_input = torch.tensor(species_query_input, dtype=torch.float)
        pathway_query_input = torch.tensor(pathway_query_input, dtype=torch.float)
        support_label = torch.tensor(support_label, dtype=torch.long)
        query_label = torch.tensor(query_label, dtype=torch.long)

        return species_support_input,pathway_support_input, \
               species_query_input,pathway_query_input, support_label, query_label

    def fit(self, species_X,pathway_X, Y, optimizer, criterion, EPOCH):
        loss_list = []
        for epoch in range(EPOCH):
            self.train()
            optimizer.zero_grad()
            species_support_input, pathway_support_input, \
            species_query_input, pathway_query_input, support_label, query_label \
                = self.randomGenerate(species_X,pathway_X, Y)

            output = self.forward(species_support_input,pathway_support_input,
                species_query_input,pathway_query_input)
            loss = criterion(output, query_label)

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            print("Epoch number:{},Current loss:{:.4f}\n".format(epoch, loss.item()))

        return loss_list


    def predict(self, specie_X_test):
        self.eval()
        specie_X_test = torch.tensor(specie_X_test, dtype=torch.float)
        species_embedding = self.species_feature_extraction(specie_X_test)
        result = torch.zeros(size=[species_embedding.shape[0], self.num_class])
        for i in range(0, species_embedding.shape[0]):
            temp_value = species_embedding[i].repeat(self.num_class, 1)
            dist_value = 0
            if self.distance == 'euclidean':
                dist_value = F.pairwise_distance(self.prototypical, temp_value, p=2)
            elif self.distance == 'cosine':
                dist_value = torch.cosine_similarity(self.prototypical, temp_value, dim=1)
                dist_value = 1 - dist_value
            result[i] = -1 * dist_value
        result = F.softmax(result, dim=1)
        pre_Y = result[:, 0] < result[:, 1]
        pre_Y = pre_Y.detach().numpy().astype(int)
        prob_Y = result[:, 1].detach().numpy()
        return pre_Y, prob_Y





if __name__ == '__main__':

    pathway_df = pd.read_csv("../../data/curatedMetagenomicData/NielsenHB_2014/NielsenHB_2014.pathcoverage.csv",
                             sep=',', index_col=0).T

    label_df = pd.read_table("../../data/curatedMetagenomicData/NielsenHB_2014/NielsenHB_2014_pData.csv",
                             sep=",", index_col=0)[['disease']]
    label_df.loc[label_df["disease"] == "healthy", "disease"] = 0
    label_df.loc[label_df["disease"] == "IBD", "disease"] = 1

    data_df1 = pathway_df.iloc[:, 2:].join(label_df).dropna()

    data_arr1 = np.array(data_df1)

    pathway_X = data_arr1[:, 2:-1].astype(np.float)

    species_df = pd.read_table("../../data/curatedMetagenomicData/NielsenHB_2014/counts/NielsenHB_2014_counts_species.csv",
                       sep=",", index_col=0).dropna(axis=1)

    log_df = species_df.apply(np.log1p, axis=1).T


    data_df2 = log_df.join(label_df).dropna()

    data_arr2 = np.array(data_df2)

    species_X = data_arr2[:, :-1].astype(np.float)
    y = data_arr2[:, -1].astype(np.int)

    acc = []
    auc = []
    f1 = []

    for i in range(3):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random.randint(0, 10 ** 9))
        cur_acc = []
        cur_auc = []
        cur_f1 = []
        for train_index, test_index in kf.split(species_X, y):

            species_X_train, species_X_test = species_X[train_index], species_X[test_index]
            pathway_X_train, pathway_X_test = pathway_X[train_index], pathway_X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            s_filter_column = np.sum((species_X_train != 0), axis=0) > species_X_train.shape[0] * 0.2
            species_X_train = species_X_train[:, s_filter_column]
            species_X_test = species_X_test[:, s_filter_column]

            p_filter_column = np.sum((pathway_X_train != 0), axis=0) > pathway_X_train.shape[0] * 0.2
            pathway_X_train = pathway_X_train[:, p_filter_column]
            pathway_X_test = pathway_X_test[:, p_filter_column]



            s_std = StandardScaler()
            s_std.fit(species_X_train,y_train)
            species_X_train = s_std.transform(species_X_train)
            species_X_test = s_std.transform(species_X_test)

            p_std = StandardScaler()
            p_std.fit(pathway_X_train,y_train)
            pathway_X_train = p_std.transform(pathway_X_train)
            pathway_X_test = p_std.transform(pathway_X_test)

            skb = SelectKBest(f_classif, k=256)
            # skb = UnivariateFilter(f_ratio_measure, select_k_best(256))
            skb.fit(pathway_X_train, y_train)
            pathway_X_train = skb.transform(pathway_X_train)
            pathway_X_test = skb.transform(pathway_X_test)

            acmp = Adaptive_Cross_Modal_PN(species_X_train.shape[1],
                                           pathway_X_train.shape[1],2,64,100,100)
            optimer = optim.Adam(acmp.parameters(), lr=0.001, weight_decay=0.001)
            criterion = nn.CrossEntropyLoss()
            acmp.fit(species_X_train,pathway_X_train,y_train,optimer,criterion,100)
            pre_y, prob_y = acmp.predict(species_X_test)


            cur_acc.append(accuracy_score(pre_y, y_test))
            cur_auc.append(roc_auc_score(y_test, prob_y))
            cur_f1.append(f1_score(y_test, pre_y))


    acc.append(np.mean(np.array(cur_acc)))
    auc.append(np.mean(np.array(cur_auc)))
    f1.append(np.mean(np.array(cur_f1)))

    print("IPrototypical Net   ACC:" + str(np.mean(np.array(acc))) + "    AUC：" + str(np.mean(np.array(auc))) +
          "    F1-micro：" + str(np.mean(np.array(f1))))

