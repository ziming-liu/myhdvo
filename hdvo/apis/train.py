import copy as cp

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)
from mmcv.runner.hooks import Fp16OptimizerHook, OptimizerHook
import mmcv
from mmcv.runner import EpochBasedRunner, Hook, IterBasedRunner, BaseRunner
from mmcv.runner.utils import get_host_info
import time
from torch.nn.utils import clip_grad
from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.fp16_utils import LossScaler, wrap_fp16_model

from ..core import (DistEvalHook, EvalHook, OmniSourceDistSamplerSeedHook,
                    OmniSourceRunner)
from ..datasets import build_dataloader, build_dataset
from ..utils import PreciseBNHook, get_root_logger
import random
import numpy as np


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    if cfg.omnisource:
        # The option can override videos_per_gpu
        train_ratio = cfg.data.get('train_ratio', [1] * len(dataset))
        omni_videos_per_gpu = cfg.data.get('omni_videos_per_gpu', None)
        if omni_videos_per_gpu is None:
            dataloader_settings = [dataloader_setting] * len(dataset)
        else:
            dataloader_settings = []
            for videos_per_gpu in omni_videos_per_gpu:
                this_setting = cp.deepcopy(dataloader_setting)
                this_setting['videos_per_gpu'] = videos_per_gpu
                dataloader_settings.append(this_setting)
        data_loaders = [
            build_dataloader(ds, **setting)
            for ds, setting in zip(dataset, dataloader_settings)
        ]

    else:
        data_loaders = [
            build_dataloader(ds, **dataloader_setting) for ds in dataset
        ]
    model = model.cuda() # is neccessary for DistShampoo
    #if cfg.apex.synced_bn:
        # using apex synced BN
    #    model = apex.parallel.convert_syncbn_model(model)
    
    #print(model)
    # build runner
    if cfg.optimizer["type"]=="shampoo": # second order optim
        import torch_optimizer as optim
        print("using shampoo 2nd order optimizer")
        cfg.optimizer.pop("type")
        optimizer = optim.Shampoo(
            model.parameters(),
            **cfg.optimizer
        )
    elif cfg.optimizer["type"]=="DistShampoo":
        from distributed_shampoo.distributed_shampoo import DistributedShampoo
        from distributed_shampoo.shampoo_utils import GraftingType
        print("using distributed shampoo optimizer (meta-version)")
        cfg.optimizer.pop("type")
        if "grafting_type" in cfg.optimizer:
            grafting_type = cfg.optimizer["grafting_type"]
            cfg.optimizer.pop("grafting_type")
        else:
            grafting_type = "SGD"
        GF = {"ADAM": GraftingType.ADAM, "SGD": GraftingType.SGD,\
              "ADAGRAD": GraftingType.ADAGRAD, "AdamW": GraftingType.ADAM,\
              "ADAM_NORMALIZED": GraftingType.ADAM_NORMALIZED,}
        opt_cfg = dict(   
                    lr=0.001,
                    betas=(0.9, 0.999),
                    epsilon=1e-12,
                    weight_decay=1e-05,
                    max_preconditioner_dim=8192,
                    precondition_frequency=100,
                    use_decoupled_weight_decay=False,
                    grafting_type=GF[grafting_type],
                    grafting_epsilon=1e-08,
                    grafting_beta2=0.999,)
        opt_cfg.update(cfg.optimizer)
        optimizer = DistributedShampoo(
                    model.parameters(),
                    **opt_cfg,
                )
    elif cfg.optimizer["type"]=="DistAdaGaussian":
        from distributed_adagaussian.distributed_adagaussian import DistributedAdaGaussian
        from distributed_adagaussian.adagaussian_utils import GraftingType
        print("using distributed shampoo optimizer (meta-version)")
        cfg.optimizer.pop("type")
        if "grafting_type" in cfg.optimizer:
            grafting_type = cfg.optimizer["grafting_type"]
            cfg.optimizer.pop("grafting_type")
        else:
            grafting_type = "SGD"
        GF = {"ADAM": GraftingType.ADAM, "SGD": GraftingType.SGD,\
              "ADAGRAD": GraftingType.ADAGRAD, "AdamW": GraftingType.ADAM,\
              "ADAM_NORMALIZED": GraftingType.ADAM_NORMALIZED,}
        opt_cfg = dict(   
                    lr=0.001,
                    betas=(0.9, 0.999),
                    epsilon=1e-12,
                    weight_decay=1e-05,
                    max_preconditioner_dim=8192,
                    precondition_frequency=100,
                    use_decoupled_weight_decay=False,
                    grafting_type=GF[grafting_type],
                    grafting_epsilon=1e-08,
                    grafting_beta2=0.999,)
        opt_cfg.update(cfg.optimizer)
        optimizer = DistributedAdaGaussian(
                    model.parameters(),
                    **opt_cfg,
                )
    elif cfg.optimizer["type"]=="apollo":
        print("using apollo 2nd order optimizer")
        import torch_optimizer as optim
        cfg.optimizer.pop("type")
        optimizer = optim.Apollo(
            model.parameters(),
            **cfg.optimizer
        )
    elif cfg.optimizer["type"]=="GaussianNewton":
        #import torch_optimizer as optim
        from ..core.optimizer.gaussian_newton import GaussianNewton
        print("using gaussian newton 2nd order optimizer")
        #cfg.optimizer.pop("type")
        optim_cfg = cfg.optimizer.copy()
        optim_cfg.pop("type")
        optimizer = GaussianNewton(
            model.parameters(),
            **optim_cfg
        )
    elif cfg.optimizer["type"]=="GaussianNewton2":
        #import torch_optimizer as optim
        from ..core.optimizer.gaussian_newton2 import GaussianNewton2
        print("using gaussian newton2 2nd order optimizer")
        #cfg.optimizer.pop("type")
        optim_cfg = cfg.optimizer.copy()
        optim_cfg.pop("type")
        optimizer = GaussianNewton2(
            model.parameters(),
            **optim_cfg
        )
    elif cfg.optimizer["type"]=="GaussianNewton3":
        #import torch_optimizer as optim
        from ..core.optimizer.gaussian_newton3 import GaussianNewton3
        print("using gaussian newton3 2nd order optimizer")
        #cfg.optimizer.pop("type")
        optim_cfg = cfg.optimizer.copy()
        optim_cfg.pop("type")
        optimizer = GaussianNewton3(
            model.parameters(),
            **optim_cfg
        )
    elif cfg.optimizer["type"]=="GaussianNewton5":
        #import torch_optimizer as optim
        from ..core.optimizer.gaussian_newton5 import GaussianNewton5
        print("using gaussian newton5 2nd order optimizer")
        #cfg.optimizer.pop("type")
        optim_cfg = cfg.optimizer.copy()
        optim_cfg.pop("type")
        optimizer = GaussianNewton5(
            model.parameters(),
            **optim_cfg
        )
    elif cfg.optimizer["type"]=="adahessian":
        #import torch_optimizer as optim
        from ..core.optimizer.adahessian import Adahessian
        print("using adahessian 2nd order optimizer")
        #cfg.optimizer.pop("type")
        optim_cfg = cfg.optimizer.copy()
        optim_cfg.pop("type")
        optimizer = Adahessian(
            model.parameters(),
            **optim_cfg
        )
    elif cfg.optimizer["type"]=="Lion":
        from ..core.optimizer.lion import Lion
        print("using LION optimizer")
        cfg.optimizer.pop("type")
        optimizer = Lion(model.parameters(), **cfg.optimizer)
        # lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0
    elif cfg.optimizer["type"]=="SGDfilter":
        print("using SGDfilter optimizer, which only updates the parameters with requires_grad=True")
        cfg.optimizer.pop("type")
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), **cfg.optimizer)
    else:
        optimizer = build_optimizer(model, cfg.optimizer)

     
    logger.info("Register Optimizer Hook...")

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    #if not validate:
    #    model = torch.compile(model,mode="max-autotune")
    #    print("torch compile model!!")

    #Runner = OmniSourceRunner if cfg.omnisource else EpochBasedRunner
    if_iter_runner=False
    if "minibatch" in cfg and cfg.minibatch:
        print("use minibatch training!")
        Runner = MinibatchRunner
        meta.update({"minibatch_per_epoch":cfg.minibatch})
    elif "runner" in cfg and cfg.runner["type"] == "IterBasedRunner":
        if_iter_runner = True
        Runner = IterBasedRunner 
    elif "runner" in cfg and cfg.runner["type"] == "FreezeBNIterBasedRunner":
        if_iter_runner = True
        Runner = FreezeBNIterBasedRunner
    else: Runner = EpochBasedRunner
    print(f"using Runner {Runner}")
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    mixed_precision = cfg.get('mixed_precision', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif 'type' in cfg.optimizer_config and cfg.optimizer_config["type"]=="OptimizerHookforGaussianNewton2":
        cfg.optimizer_config.pop("type")
        optimizer_config = OptimizerHookforGaussianNewton2(
            **cfg.optimizer_config, distributed=distributed)
    elif 'type' in cfg.optimizer_config and cfg.optimizer_config["type"]=="OptimizerHookforGaussianNewtonSqr":
        cfg.optimizer_config.pop("type")
        optimizer_config = OptimizerHookforGaussianNewtonSqr(
            **cfg.optimizer_config, distributed=distributed)
    elif 'type' in cfg.optimizer_config and cfg.optimizer_config["type"]=="OptimizerHookforDistAdaGaussian":
        cfg.optimizer_config.pop("type")
        optimizer_config = OptimizerHookforDistAdaGaussian(
            **cfg.optimizer_config, distributed=distributed)
    elif 'type' in cfg.optimizer_config and cfg.optimizer_config["type"]=="Fp16OptimizerHookforAdaHessian":
        cfg.optimizer_config.pop("type")
        optimizer_config = Fp16OptimizerHookforAdaHessian(
            **cfg.optimizer_config, distributed=distributed)
    elif 'type' in cfg.optimizer_config and cfg.optimizer_config["type"]=="OptimizerHookforAdaHessian":
        cfg.optimizer_config.pop("type")
        optimizer_config = OptimizerHookforAdaHessian(
            **cfg.optimizer_config,)
    elif mixed_precision is not None:
        print("## using mixed precision training!")
        optimizer_config = MixedPrecisionOptimizerHook(
            **cfg.optimizer_config,)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed and not if_iter_runner:
        if cfg.omnisource:
            runner.register_hook(OmniSourceDistSamplerSeedHook())
        else:
            runner.register_hook(DistSamplerSeedHook())

    # precise bn setting
    if cfg.get('precise_bn', False):
        precise_bn_dataset = build_dataset(cfg.data.train)
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=0,  # save memory and time
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed)
        data_loader_precise_bn = build_dataloader(precise_bn_dataset,
                                                  **dataloader_setting)
        precise_bn_hook = PreciseBNHook(data_loader_precise_bn,
                                        **cfg.get('precise_bn'))
        runner.register_hook(precise_bn_hook)

    if validate:
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
    
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()
    if cfg.omnisource:
        runner_kwargs = dict(train_ratio=train_ratio)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, **runner_kwargs)

        


class MinibatchRunner(EpochBasedRunner):
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            if i%self.meta["minibatch_per_epoch"] ==0 and i!=0:
                break
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1


class FreezeBNIterBasedRunner(IterBasedRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.model.module.freeze_bn() # keep BatchNorm frozen
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1


try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

class MixedPrecisionOptimizerHook(OptimizerHook):
    
    def __init__(self, grad_clip=None, **kwargs):
        super().__init__(grad_clip=grad_clip, **kwargs)
        self.grad_clip = grad_clip
        self.scaler = GradScaler(enabled=True)

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        self.scaler.scale(runner.outputs['loss']).backward()
        self.scaler.unscale_(runner.optimizer)
        #runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        #runner.optimizer.step()
        self.scaler.step(runner.optimizer)
        self.scaler.update()

class Fp16OptimizerHookforAdaHessian(Fp16OptimizerHook):
    def __init__(self,
                    grad_clip=None,
                    coalesce=True,
                    bucket_size_mb=-1,
                    loss_scale=512.,
                    distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        if loss_scale == 'dynamic':
            self.loss_scaler = LossScaler(mode='dynamic')
        elif isinstance(loss_scale, float):
            self.loss_scaler = LossScaler(
                init_scale=loss_scale, mode='static')
        elif isinstance(loss_scale, dict):
            self.loss_scaler = LossScaler(**loss_scale)
        else:
            raise ValueError('loss_scale must be of type float, dict, or '
                                f'"dynamic", got {loss_scale}')

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer `loss_scalar.py`

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients (fp16).
        3. Copy gradients from the model to the fp32 weight copy.
        4. Scale the gradients back and update the fp32 weight copy.
        5. Copy back the params from fp32 weight copy to the fp16 model.
        6. Save loss_scaler state_dict for resume purpose.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        # scale the loss value
        scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
        scaled_loss.backward(retain_graph=True, create_graph = True)
        # copy fp16 grads in the model to fp32 params in the optimizer

        fp32_weights = []
        for param_group in runner.optimizer.param_groups:
            fp32_weights = fp32_weights + param_group['params']
        self.copy_grads_to_fp32(runner.model, fp32_weights)
        # allreduce grads
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce,
                            self.bucket_size_mb)

        has_overflow = self.loss_scaler.has_overflow(fp32_weights)
        # if has overflow, skip this iteration
        if not has_overflow:
            # scale the gradients back
            for param in fp32_weights:
                if param.grad is not None:
                    param.grad.div_(self.loss_scaler.loss_scale)
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(fp32_weights)
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update(
                        {'grad_norm': float(grad_norm)},
                        runner.outputs['num_samples'])
            # update fp32 params
            runner.optimizer.step()
            # copy fp32 params to the fp16 model
            self.copy_params_to_fp16(runner.model, fp32_weights)
        self.loss_scaler.update_scale(has_overflow)
        if has_overflow:
            runner.logger.warning('Check overflow, downscale loss scale '
                                    f'to {self.loss_scaler.cur_scale}')

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()


class OptimizerHookforAdaHessian(OptimizerHook):
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward(create_graph = True)
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()

class OptimizerHookforGaussianNewton2(OptimizerHook):
    def __init__(self, grad_clip=None, distributed=True, **kwargs):
        super().__init__(grad_clip=grad_clip, **kwargs)
        self.grad_clip = grad_clip

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['fxloss'] = torch.sqrt(2*runner.outputs['loss'])
        runner.outputs['fxloss'].backward(retain_graph=True)
        l1_grads = [p.grad.detach().clone()   for p in runner.model.parameters() if (p.requires_grad is True and p.grad is not None)]
        #runner.outputs["l2_loss"] = 0.5*runner.outputs['loss']**2
        runner.outputs["loss"].backward(retain_graph=True)
        l2_grads = [p.grad.detach().clone()   for p in runner.model.parameters() if (p.requires_grad is True and p.grad is not None)]
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step(l1_grads=l1_grads, l2_grads=l2_grads)

class OptimizerHookforGaussianNewtonSqr(OptimizerHook):
    def __init__(self, grad_clip=None, distributed=True, **kwargs):
        super().__init__(grad_clip=grad_clip, **kwargs)
        self.grad_clip = grad_clip

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        #runner.outputs['loss'] = torch.sqrt(2*runner.outputs['loss'])
        runner.outputs['loss'].backward(retain_graph=True)
        l1_grads = [p.grad.detach().clone()   for p in runner.model.parameters() if (p.requires_grad is True and p.grad is not None)]
        runner.outputs["l2_loss"] = 0.5*runner.outputs['loss']**2
        runner.outputs["l2_loss"].backward(retain_graph=True)
        l2_grads = [p.grad.detach().clone()   for p in runner.model.parameters() if (p.requires_grad is True and p.grad is not None)]
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step(l1_grads=l1_grads, l2_grads=l2_grads)


class OptimizerHookforDistAdaGaussian(OptimizerHook):
    def __init__(self, grad_clip=None, distributed=True, **kwargs):
        super().__init__(grad_clip=grad_clip, **kwargs)
        self.grad_clip = grad_clip

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        
        #runner.outputs["l2_loss"] = 0.5*runner.outputs['loss']**2
        runner.outputs["loss"].backward(retain_graph=True)
        l2_grads = [p.grad.detach().clone()   for p in runner.model.parameters() if (p.requires_grad is True )]
        
        runner.outputs['fxloss'] = torch.sqrt(2*runner.outputs['loss'])
        runner.outputs['fxloss'].backward(retain_graph=True)
        #l1_grads = [p.grad.detach().clone()   for p in runner.model.parameters() if (p.requires_grad is True and p.grad is not None)]
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step(l2_grads=l2_grads)