# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import optim
import matplotlib.pyplot as plt

from Utils.utils import mixup_data

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

class PrototypicalNet(nn.Module):
    def __init__(self,in_feature, num_class, embedding_dim,
                 support_num,query_num ,distance='euclidean',mixup_data = False):
        super(PrototypicalNet, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.support_num = support_num
        self.query_num = query_num
        self.distance = distance
        self.prototypical = None
        self.prototypes = []
        self.mixup_data = mixup_data
        self.feature_extraction = nn.Sequential(

            nn.Linear(in_features=in_feature, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=embedding_dim),

        )


        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1, )
                torch.nn.init.constant_(m.bias, 0)


    def embedding(self, features):
        result = self.feature_extraction(features)
        return result

    def forward(self, support_input, query_input):

        support_embedding = self.embedding(support_input)
        query_embedding = self.embedding(query_input)
        support_size = support_embedding.shape[0]

        every_class_num = support_size // self.num_class

        class_meta_dict = {}
        for i in range(0, self.num_class):
            ### 0 : pos
            class_meta_dict[i] = torch.sum(support_embedding[i * every_class_num:(i + 1) * every_class_num, :],
                                           dim=0) / every_class_num

        class_meta_information = torch.zeros(size=[len(class_meta_dict), support_embedding.shape[1]])
        for key, item in class_meta_dict.items():
            class_meta_information[key, :] = class_meta_dict[key]

        N_query = query_embedding.shape[0]
        result = torch.zeros(size=[N_query, self.num_class])

        self.prototypical = class_meta_information
        self.prototypes.append(class_meta_information.detach().numpy())

        for i in range(0, N_query):
            temp_value = query_embedding[i].repeat(self.num_class, 1)
            dist_value = 0

            # de = torch.pow(torch.norm(self.prototypical - temp_value,dim=1),2)
            # dn = torch.pow(torch.norm(self.prototypical,dim=1) -
            #                torch.norm(temp_value,dim=1),2)
            #
            # result[i] = torch.sqrt(de+dn)

            if self.distance == 'euclidean':
                dist_value = F.pairwise_distance(self.prototypical, temp_value, p=2)
            elif self.distance == 'cosine':
                dist_value = torch.cosine_similarity(self.prototypical, temp_value, dim=1)
                dist_value = 1 - dist_value
            result[i] = -1 * dist_value

        return result




    def randomGenerate(self,X,Y):
        # N = X.shape[0]  ### 所有样本的个数
        ###取正负样本  2/3 做支持   1/3做查询
        postive_num = (Y == 1).sum()  ###正样本个数

        negtive_num = (Y == 0).sum()  ###负样本个数
        postive_index = np.where(Y == 1)[0]
        negtive_index = np.where(Y == 0)[0]


        pos_support_index = np.random.choice(postive_index, self.support_num // 2, replace=False)

        neg_support_index = np.random.choice(negtive_index, self.support_num // 2, replace=False)

        support_index = np.concatenate((neg_support_index,pos_support_index), axis=0)
        support_input = X[support_index, :]
        support_label = Y[support_index]


        pos_query_index = np.random.choice([index for index in postive_index if index not in pos_support_index],
                                           self.query_num // 2, replace=False)
        neg_query_index = np.random.choice([index for index in negtive_index if index not in neg_support_index],
                                           self.query_num // 2, replace=False)
        query_index = np.concatenate((neg_query_index,pos_query_index), axis=0)
        query_input = X[query_index]
        query_label = Y[query_index]

        support_input = torch.tensor(support_input, dtype=torch.float)
        query_input = torch.tensor(query_input, dtype=torch.float)
        support_label = torch.tensor(support_label, dtype=torch.long)
        query_label = torch.tensor(query_label, dtype=torch.long)

        return support_input, query_input, support_label, query_label


    def fit(self,X,Y,optimizer,criterion,EPOCH):
        loss_list = []
        for epoch in range(EPOCH):
            self.train()
            optimizer.zero_grad()
            support_input, query_input, support_label, query_label = self.\
                randomGenerate(X,Y)

            if not self.mixup_data:
                output = self.forward(support_input, query_input)
                loss = criterion(output, query_label)
            else:
                mixed_query_input, query_label_a, query_label_b, lam = mixup_data(query_input, query_label)
                out = self.forward(support_input, mixed_query_input)
                loss = lam * criterion(out, query_label_a) + (1 - lam) * criterion(out, query_label_b)

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            print("Epoch number:{},Current loss:{:.6f}\n".format(epoch, loss.item()))

        return loss_list


    def predict(self, X_test):
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_embedding = self.embedding(X_test)
        result = torch.zeros(size=[X_embedding.shape[0], self.num_class])
        for i in range(0, X_embedding.shape[0]):
            temp_value = X_embedding[i].repeat(self.num_class, 1)
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








def protoNet_visualie(model, X, Y, disease_name,path):
    prototypical = model.prototypical.detach().numpy()

    X = torch.tensor(X, dtype=torch.float32)
    X_transform = model.embedding(X)
    postive_index = np.where(Y == 1)[0]
    negtive_index = np.where(Y == 0)[0]
    X_0 = X_transform.detach().numpy()[negtive_index, :]
    X_1 = X_transform.detach().numpy()[postive_index, :]
    plt.figure()
    plt.scatter(X_0[:, 0], X_0[:, 1], c='#FF9900', label='Control')
    plt.scatter(prototypical[0, 0], prototypical[0, 1], marker='*', color='black', s=80, label='Control Prototype')
    plt.scatter(X_1[:, 0], X_1[:, 1], c='#006699', label=disease_name)
    plt.scatter(prototypical[1, 0], prototypical[1, 1], marker='X', color='red', s=80,
                label=disease_name + ' Prototype')

    plt.title("ProtoTypicalNet Output Visualization", )
    plt.legend()
    plt.savefig(path, dpi=400)
    plt.show()





if __name__ == '__main__':

    df = pd.read_table("../data/curatedMetagenomicData/JieZ_2017/counts/JieZ_2017_counts_species.csv",
                       sep=",", index_col=0)
    df = df.dropna(axis=1)

    flag_df = (df != 0).sum(axis=1) > df.shape[1] * 0.2
    filted_df = df[flag_df]
    log_df = filted_df.apply(np.log1p, axis=1).T
    label_df = pd.read_table("../data/curatedMetagenomicData/JieZ_2017/JieZ_2017_pData.csv",
                             sep=",", index_col=0)[['disease']]

    label_df = label_df.dropna()
    label_df.loc[label_df["disease"] == "healthy", "disease"] = 0
    label_df.loc[label_df["disease"] == "ACVD", "disease"] = 1

    data_df = log_df.join(label_df)
    data_df = data_df.dropna()
    data_arr = np.array(data_df)
    X = data_arr[:, :-1].astype(float)
    Y = data_arr[:, -1].astype(float)

    from sklearn.preprocessing import StandardScaler


    print(X.shape)
    print(Y.sum())

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y ,test_size=0.3,
                                                        random_state=1234)
    std = StandardScaler()
    std.fit(X_train)
    X_train = std.transform(X_train)
    X_test = std.transform(X_test)

    model = PrototypicalNet(in_feature=X_train.shape[1], num_class=2,embedding_dim=2,
                            support_num=120,query_num=80)
    optimer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    model.fit(X_train,Y_train,optimer,criterion,100)
    pre_y ,_ = model.predict(X_test)
    print(accuracy_score(Y_test, pre_y))


