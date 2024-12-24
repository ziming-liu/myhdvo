import torch
import torch.nn as nn

from ..registry import LOSSES


def mse_loss(reg_score,
            label,
            threshold=0.5,
            ratio_range=(1.05, 21),
            eps=1e-5):
    """Binary Logistic Regression Loss."""
    #print(label[0])
    #print(reg_score[0])
    #exit()
    # qianzhonghou
    #y = label[:,int(label.size(1)/2-1),:].view(-1, 6).to(reg_score.device)
    y = label[:,-1,:].view(-1, 6).to(reg_score.device)
    predicted = reg_score.contiguous().view(-1, 6)
    angle_loss = torch.nn.functional.mse_loss(predicted[:,:3], y[:,:3])
    translation_loss = torch.nn.functional.mse_loss(predicted[:,3:], y[:,3:])
    loss = (100 * angle_loss + 1 * translation_loss)
    loss = torch.mean(loss)
    return loss


@LOSSES.register_module()
class MSELoss(nn.Module):
    """Binary Logistic Regression Loss.

    It will calculate binary logistic regression loss given reg_score and
    label.
    """

    def forward(self,
                reg_score,
                label,
                threshold=0.5,
                ratio_range=(1.05, 21),
                eps=1e-5):
        """Calculate Binary Logistic Regression Loss.

        Args:
                reg_score (torch.Tensor): Predicted score by model.
                label (torch.Tensor): Groundtruth labels.
                threshold (float): Threshold for positive instances.
                    Default: 0.5.
                ratio_range (tuple): Lower bound and upper bound for ratio.
                    Default: (1.05, 21)
                eps (float): Epsilon for small value. Default: 1e-5.

        Returns:
                torch.Tensor: Returned binary logistic loss.
        """

        return mse_loss(reg_score, label, threshold,
                                               ratio_range, eps)
