'''
Date: 2021-03-02 22:38:16
LastEditors: Ziming Liu
LastEditTime: 2023-10-07 20:29:30
Description: ...
Dependent packages: don't need any extral dependency
'''
from .base import BaseWeightedLoss
from .binary_logistic_regression_loss import BinaryLogisticRegressionLoss
from .cross_entropy_loss import BCELossWithLogits, CrossEntropyLoss
from .mse_loss import MSELoss
#from .zncc_loss import ZNCC,ZNCCLoss
from .zncc_lossc1 import ZNCC,ZNCCLoss
from .l1_loss import L1Loss
from .l2_loss import L2Loss
from .ssim_loss import SSIMLoss
from .disp_smooth_loss import DispSmoothLoss
from .disp_l1_loss import DispL1Loss
from .sigloss import SigLoss,SiLogLoss
from .chamferloss import BinsChamferLoss
from .celoss import CrossEntropyLoss2
from .disp_ce_loss import DispCELoss
from .huber_loss import HUBERLoss