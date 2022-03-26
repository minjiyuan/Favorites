
import torch.nn as nn


"""
PNTriletloss is the implementation of the mixed loss in the paper

"""

class PNTripletloss(nn.Module):

    def __init__(self,l=0.1,margin = 5):
        super(PNTripletloss, self).__init__()
        self.l = l
        self.margin = margin

    def forward(self,y_pred,y):
        loss1 = nn.CrossEntropyLoss().forward(y_pred,y)
        loss2 = 0
        for i in range(y_pred.shape[0]):
            if y[i] == 0:
                loss2 += max(-1 * y_pred[i][0] - (-1 * y_pred[i][1]) + self.margin, 0)
            else:
                loss2 += max(-1 * y_pred[i][1] - (-1 * y_pred[i][0]) + self.margin, 0)


        loss2 = loss2 / y_pred.shape[0]
        loss = loss1 + self.l * loss2

        return loss


