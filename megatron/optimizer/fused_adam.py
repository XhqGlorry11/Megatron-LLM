import torch
from apex.multi_tensor_apply import multi_tensor_applier
from torch.optim.optimizer import _dispatch_sqrt

class FusedAdam(torch.optim.Optimizer):

    """Implements Adam algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedAdam` may be used with or without Amp.  If you wish to use :class:`FusedAdam` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.


    .. warning::
        A previous version of :class:`FusedAdam` allowed a number of additional arguments to ``step``.  These additional arguments
        are now deprecated and unnecessary.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with CUDA Graphs. (default: False)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16 mixed precision training, currently can
           only be used with capturable set to True. (default: False)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, adam_w_mode=True,
                 weight_decay=0., amsgrad=False, set_grad_none=True,
                 capturable=False, master_weights=False, diasble_zero_grad_parameter_update=False):
        self.diasble_zero_grad_parameter_update = diasble_zero_grad_parameter_update
        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        if master_weights and not capturable:
            raise RuntimeError('Master weights is currently only supported with the capturable version.')
        # If the optimizer is capturable then LR should be a tensor (on GPU)
        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none

        self.capturable = capturable
        self.master_weights = master_weights

        # Create full precision master weights
        self.param_groups_master = []
        for i, pg in enumerate(self.param_groups):
            param_list = pg['params']
            self.param_groups_master.append({
                'params': [
                    p.clone().detach().float() if self.master_weights else None
                    for p in param_list
                ],
            })

        if capturable:
            for idx, group in enumerate(self.param_groups):
                if len(group['params']) == 0:
                    continue
                device = group['params'][0].device
                for item in ['lr']:
                    self.param_groups[idx][item] = group[item].to(device=device)

            self._step_supports_amp_scaling = True

        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
            self.multi_tensor_adam_capturable = amp_C.multi_tensor_adam_capturable
            self.multi_tensor_adam_capturable_master = amp_C.multi_tensor_adam_capturable_master
        else:
            raise RuntimeError('apex.optimizers.FusedAdam requires cuda extensions')

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedAdam, self).zero_grad()

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError('FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.')
        loss = None
        if closure is not None:
            loss = closure()
        # np.save('/home/xinghq/npy/p2.npy', self.param_groups[0]['params'][2].detach().cpu().numpy())
        # parameter = self.param_groups[0]['params'][2]
        # cur_s = self.state[parameter]
        # np.save('/home/xinghq/npy/avg_sq2.npy', cur_s['exp_avg_sq'].detach().cpu().numpy())
        for group, group_master in zip(self.param_groups, self.param_groups_master):
            if len(group['params']) == 0:
                continue
            device = group['params'][0].device
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1 if not self.capturable else (self._dummy_overflow_buf != 1).to(torch.int)
            else:
                group['step'] = 1 if not self.capturable else torch.tensor([1], dtype=torch.int, device=device)

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            p_16_master = []
            p_32_master = []

            for p, p_master in zip(group['params'], group_master['params']):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()

                if p.dtype == torch.float16:
                    if self.master_weights:
                        p_16_master.append(p_master.data)
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.bfloat16:
                    g_bf.append(p.grad)
                    p_bf.append(p)
                    m_bf.append(state['exp_avg'])
                    v_bf.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    if self.master_weights:
                        p_32_master.append(p_master.data)
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedAdam only support fp16 and fp32.')

            # If the optimizer is capturable, then if there's a grad scaler it works
            # on the GPU + a different multi_tensor_applier should be called
            if self.capturable:
                # overflow check of gradients
                found_inf = (
                    grad_scaler._check_inf_per_device(self)[device]
                    if grad_scaler is not None else torch.zeros((1,), device=device)
                )
                self._dummy_overflow_buf.copy_(found_inf)

                # get unscale scale factor
                scale, inv_scale = None, None
                if grad_scaler:
                    scale = grad_scaler._get_scale_async()
                    inv_scale = scale.double().reciprocal().float()
                else:
                    scale = torch.ones((1,), device=device)
                    inv_scale = torch.ones((1,), device=device)

                if len(g_16) > 0:
                    multi_tensor_applier(self.multi_tensor_adam_capturable_master if self.master_weights
                            else self.multi_tensor_adam_capturable,
                            self._dummy_overflow_buf,
                            [g_16, p_16, m_16, v_16, p_16_master] if self.master_weights
                            else [g_16, p_16, m_16, v_16],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'],
                            inv_scale)

                if len(g_bf) > 0:
                    multi_tensor_applier(
                            self.multi_tensor_adam_capturable,
                            self._dummy_overflow_buf,
                            [g_bf, p_bf, m_bf, v_bf],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'],
                            inv_scale)

                if len(g_32) > 0:
                    multi_tensor_applier(self.multi_tensor_adam_capturable_master if self.master_weights
                            else self.multi_tensor_adam_capturable,
                            self._dummy_overflow_buf,
                            [g_32, p_32, m_32, v_32, p_32_master] if self.master_weights
                            else [g_32, p_32, m_32, v_32],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'],
                            inv_scale)
            else:
                if len(g_16) > 0:
                    multi_tensor_applier(self.multi_tensor_adam,
                            self._dummy_overflow_buf,
                            [g_16, p_16, m_16, v_16],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'])

                if len(g_bf) > 0:
                    multi_tensor_applier(
                            self.multi_tensor_adam,
                            self._dummy_overflow_buf,
                            [g_bf, p_bf, m_bf, v_bf],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'])
                
                if len(g_32) > 0:
                    if not self.diasble_zero_grad_parameter_update:
                        multi_tensor_applier(self.multi_tensor_adam,
                                self._dummy_overflow_buf,
                                [g_32, p_32, m_32, v_32],
                                group['lr'],
                                beta1,
                                beta2,
                                group['eps'],
                                group['step'],
                                self.adam_w_mode,
                                bias_correction,
                                group['weight_decay'])
                    else:
                        for param_index in range(len(p_32)):
                            param = p_32[param_index]
                            grad = g_32[param_index]
                            grad_nonzero_mask = torch.zeros_like(grad)
                            non_zero_index = torch.where(grad != 0)
                            grad_nonzero_mask[non_zero_index] = 1
                            exp_avg = m_32[param_index]
                            exp_avg_sq = v_32[param_index]
                            # weight decay
                            param.mul_(1 - group['lr'] * group['weight_decay'] * param * grad_nonzero_mask)

                            # get the first order moment running average 
                            beta1_mask = torch.ones_like(grad)
                            beta1_mask[non_zero_index] = beta1
                            exp_avg.lerp_(grad, 1 - beta1_mask)
                            # get the second order moment running average
                            beta2_mask = torch.ones_like(grad)
                            beta2_mask[non_zero_index] = beta2
                            exp_avg_sq.mul_(beta2_mask).addcmul_(grad, grad * (1 - beta2_mask), value=1)

                            bias_correction1 = 1 - beta1 ** group['step']
                            bias_correction2 = 1 - beta2 ** group['step']
                            step_size = group['lr'] / bias_correction1
                            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
                            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                            param.addcdiv_(exp_avg * grad_nonzero_mask, denom, value=-step_size)

                # import numpy as np
                # np.save('/home/xinghq/npy/grad.npy', g_32[2].detach().cpu().numpy())
                # np.save('/home/xinghq/npy/param.npy', p_32[2].detach().cpu().numpy())
                # np.save('/home/xinghq/npy/avg.npy', m_32[2].detach().cpu().numpy())
                # np.save('/home/xinghq/npy/avg_sq.npy', v_32[2].detach().cpu().numpy())
                

        return loss
