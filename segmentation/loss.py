from monai.losses.dice import DiceLoss
from monai.losses import DiceFocalLoss, DiceCELoss, FocalLoss, GeneralizedDiceLoss, GeneralizedWassersteinDiceLoss
import numpy as np
import torch
from torch import nn


def tversky(pred, target, alpha = 0.7):

    output=list()
    
    for i, j in zip(pred,target):
    
        if len(i.shape)==4 and len(j.shape)==4:
            true_pos = (i * j).sum(dim=1).sum(dim=1).sum(dim=1)
            false_neg = (j * (1-i)).sum(dim=1).sum(dim=1).sum(dim=1)
            false_pos = ((1-j)*i).sum(dim=1).sum(dim=1).sum(dim=1)
        elif len(i.shape)==3 and len(j.shape)==3:
            true_pos = (i * j).sum(dim=1).sum(dim=1)
            false_neg = (j * (1-i)).sum(dim=1).sum(dim=1)
            false_pos = ((1-j)*i).sum(dim=1).sum(dim=1)
        
        res = (true_pos + 1e-5)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + 1e-5)
        output.append(res)
        
    output=torch.stack(output, dim=0)

    return torch.mean(output) #np.mean(output)


class Tversky_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, alpha=0.7):

        #pred = pred.squeeze(dim=1)
        
        t_loss = tversky(pred, target, alpha)
        
        final_loss = 1-t_loss

        return final_loss


losses_options = {'soft_dice': DiceLoss(reduction='none'),
                  'generalized_dice': GeneralizedDiceLoss(), 
                  'cross_entropy': nn.CrossEntropyLoss(), 
                  'focal': FocalLoss(), 
                  'softdice&focal': DiceFocalLoss(), 
                  'dice&sce': DiceCELoss(),
                  'tversky': Tversky_loss()}


class Loss_function(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        
        self.first_loss = losses_options[loss_type]
        self.dice_loss = GeneralizedDiceLoss(reduction='none') #DiceLoss before

    def forward(self, pred, target):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        target = target.to(device)
        
        pred_first = pred[0]
        pred_second = pred[1]
        pred_third = pred[2]
        
        L1 = self.first_loss(pred_first, target)
        L2 = self.dice_loss(pred_second, target)
        L3 = self.dice_loss(pred_third, target)

        final_loss = L1+L2+L3

        return final_loss
        
        
class Loss_function_lynph(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        
        #self.first_loss = DiceLoss(to_onehot_y=False,reduction="mean", sigmoid=True) #losses_options[loss_type]
        #self.dice_loss = DiceLoss(to_onehot_y=False,reduction="mean", sigmoid=True)
        
        self.first_loss = DiceCELoss(to_onehot_y=False,reduction="mean", sigmoid=True) #losses_options[loss_type]
        self.dice_loss = GeneralizedDiceLoss(to_onehot_y=False,reduction="mean", sigmoid=True)

    def forward(self, pred, target):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        target = target.to(device)
        
        pred_first = pred[0]
        pred_second = pred[1]
        pred_third = pred[2]
        
        
        L1 = self.first_loss(pred_first, target)
        L2 = self.dice_loss(pred_second, target)
        L3 = self.dice_loss(pred_third, target)

        final_loss = L1+L2+L3

        return final_loss
