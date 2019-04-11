import torch.nn as nn
import torch.nn.functional as F
import torch
__all__=["CPNLoss"]

class CPNLoss(nn.Module):
    def __init__(self,num_kps=14):
        super(CPNLoss,self).__init__()
    def ohem(self,loss, top_k, batch_size):
        pass
    def l2_loss(self,global_preds, 
            refine_preds, targets, vis_mask):
        b,c,h,w = global_preds.size()
        # global loss , only calculate visuable keypoints
        loss_global = F.mse_loss(global_preds,targets)/2
        #refine loss, only calculate visuable keypoints
        loss_refine = F.mse_loss(refine_preds,targets)/2
        # loss
        loss = loss_global * 0.001 + loss_refine
        return loss,loss_global,loss_refine
    def l1_loss(self,global_preds, 
            refine_preds, targets, vis_mask):
        pass
    def forward(self,global_preds, 
            refine_preds, targets, vis_mask):
        loss,loss_global,loss_refine = \
            self.l2_loss(global_preds,refine_preds,targets,vis_mask)
        return loss,loss_global,loss_refine