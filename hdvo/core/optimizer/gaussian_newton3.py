import math
import torch
from torch.optim.optimizer import Optimizer
import  torch.distributed as dist
#import matplotlib.pyplot as plt

class GaussianNewton3(Optimizer):
    """Implements GaussianNewton3 algorithm. J*J * J_l 
    It has been proposed in `GaussianNewton3: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1). You can also try 0.5. For some tasks we found this to result in better performance.
        single_gpu (Bool, optional): Do you use distributed training or not "torch.nn.parallel.DistributedDataParallel" (default: True)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1, single_gpu=True, lam=0, 
                 spatial_average=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)
        self.single_gpu = single_gpu 
        self.lam = lam
        self.spatial_average = spatial_average
        super(GaussianNewton3, self).__init__(params, defaults)

    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """
        
         
        diagnol = []
        for g in grads:
            param_size = g.size()
            doG = g**2 # diagonal of G hessian matrix 
            #jj = g.reshape((-1,1)) @ g.reshape((-1,1)).T 
            #plt.imshow(jj.detach().cpu().numpy(), cmap='viridis')  # 使用'viridis'颜色映射，您可以选择其他映射
            #plt.colorbar()  # 添加颜色条
            #plt.title('矩阵热力图')  # 添加标题
            #plt.savefig('/home/ziliu/vis/JJheatmap.png', format='png')  # 将文件名替换为您想要的文件名和格式

            #plt.show()  # 显示图像

            if self.spatial_average:
                # for size <= 3D  Hessian diagonal block size is 1.
                if len(param_size) ==4: # conv kernel
                    # Hessian diagonal block size is 9 here: torch.sum() reduces
                    # the dim 2/3.
                    doG = torch.mean(doG, dim=[2,3], keepdim=True)
                elif len(param_size) == 5:
                    doG = torch.mean(doG, dim=[2,3,4], keepdim=True)
            if self.lam > 0:
                doG = doG + self.lam * torch.ones_like(doG)
            diagnol.append(doG)

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for output1 in diagnol:
                dist.all_reduce(output1 / torch.cuda.device_count())
        
        return diagnol

    def step(self, l1_grads=None, l2_grads=None, step_loss=None, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        params = []
        groups = []
        grads = []

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads 
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        
        # get the Hessian diagonal

        hut_traces = self.get_trace(params, l1_grads)

        for (p, group, l2_grad, hut_trace) in zip(params, groups, l2_grads, hut_traces):

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of Hessian diagonal square values
                state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

            beta1, beta2 = group['betas']

            state['step'] += 1

            # Decay the first and second moment running average coefficient
            #print("step loss ", step_loss)
            #B = grad * step_loss
            #print("B ", B)
            exp_avg.mul_(beta1).add_(l2_grad.detach_(), alpha=1 - beta1)
            exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            # make the square root, and the Hessian power
            k = group['hessian_power']
            denom = (
                (exp_hessian_diag_sq.sqrt() ** k) /
                math.sqrt(bias_correction2) ** k).add_(
                group['eps'])

            # make update
            p.data = p.data - \
                group['lr'] * (exp_avg / bias_correction1 / denom + group['weight_decay'] * p.data)

        return loss