""" 
Only contains the focal loss implementation for action bin loss.
For residuals the MSE loss can be used directly.
"""
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Computed the focal loss for the action bin classification"""
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class MultiTaskLoss(nn.Module):
    """
    This MT-loss suggested in the paper, the MSE between predicted and GT residuals for the GT bin is computed and averaged across batch and sequence.
    """
    def __init__(self, size_average: bool = True):
        super(MultiTaskLoss, self).__init__()
        self.size_average = size_average
        self.mse = nn.MSELoss()
    
    def forward(self, input, target_bins, target_residuals):
        input = input.gather(
                                1,  
                                target_bins.unsqueeze(-1).expand(-1, -1, input.size(-1))
                            ).squeeze(1)
        loss = self.mse(input, target_residuals.reshape(-1,target_residuals.shape[-1]))
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()