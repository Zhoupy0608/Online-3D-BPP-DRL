import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from acktr.utils import AddBias
from acktr.error_handler import MultiProcessLogger, GradientChecker, DeviceErrorDetector

# TODO: In order to make this code faster:
# 1) Implement _extract_patches as a single cuda kernel
# 2) Compute QR decomposition in a separate process
# 3) Actually make a general KFAC optimizer so it fits PyTorch


def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def compute_cov_a(a, classname, layer_info, fast_cnn):
    batch_size = a.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
    elif classname == 'AddBias':
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()

    return a.t() @ (a / batch_size)


def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == 'AddBias':
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size
    return g_.t() @ (g_ / g.size(0))


def update_running_stat(aa, m_aa, momentum):
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa += aa
    m_aa *= (1 - momentum)


class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x

# def check_nan(model,index):
#     for p in model.parameters():
#         if np.isnan(p.grad.data.mean().item()):
#             print('index '+ str(index) +' happened an error!')

class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.25,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=10):
        defaults = dict()

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias') and child.bias is not None:
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        split_bias(model)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}

        self.momentum = momentum
        self.stat_decay = stat_decay

        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf
        
        # Flag to control Fisher statistics accumulation
        self.acc_stats = False

        self.optim = optim.SGD(
            model.parameters(),
            lr=self.lr * (1 - self.momentum),
            momentum=self.momentum)
        
        # Initialize error handling components (Requirement 8.1, 8.4, 8.5)
        self.logger = MultiProcessLogger(name='KFAC')
        self.gradient_checker = GradientChecker(logger=self.logger)
        self.nan_inf_count = 0
        self.device_mismatch_count = 0

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            aa = compute_cov_a(input[0].data, classname, layer_info,
                               self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            gg = compute_cov_g(grad_output[0].data, classname, layer_info,
                               self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
                                    "You must have a bias as a separate layer"

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self):
        """
        Perform KFAC optimization step with comprehensive error handling.
        
        Requirements 8.1, 8.4, 8.5: Error handling with process ID logging,
        gradient checking, and graceful fallback to SGD on errors.
        """
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)

        # Requirement 8.4: Check for NaN/Inf in gradients before processing
        has_nan, has_inf, problematic_params = self.gradient_checker.check_gradients(
            self.model, self.steps
        )
        
        # If NaN/Inf detected, skip KFAC update and use SGD
        if has_nan or has_inf:
            self.nan_inf_count += 1
            self.logger.warning(
                f"NaN/Inf detected in gradients at step {self.steps} "
                f"(total occurrences: {self.nan_inf_count}). "
                f"Falling back to SGD with gradient clipping."
            )
            # Clip gradients to prevent further issues
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.steps += 1
            return

        updates = {}
        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())
                       ) == 1, "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                # Compute eigenvalue decomposition for Fisher matrices
                # This accumulates statistics from all processes via the running averages
                try:
                    self.d_g[m], self.Q_g[m] = torch.linalg.eigh(
                        self.m_gg[m])
                    self.d_a[m], self.Q_a[m] = torch.linalg.eigh(
                        self.m_aa[m])
                except RuntimeError as e:
                    self.logger.error(
                        f"Eigenvalue decomposition failed at step {self.steps}: {e}. "
                        f"Falling back to SGD for this step."
                    )
                    self.optim.step()
                    self.steps += 1
                    return

                # Requirement 8.4: Eigenvalue thresholding for numerical stability
                # Set eigenvalues < 1e-6 to 0 to prevent division by very small numbers
                num_thresholded_a = (self.d_a[m] < 1e-6).sum().item()
                num_thresholded_g = (self.d_g[m] < 1e-6).sum().item()
                
                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())
                
                if num_thresholded_a > 0 or num_thresholded_g > 0:
                    self.logger.debug(
                        f"Step {self.steps}: Thresholded {num_thresholded_a} eigenvalues in d_a "
                        f"and {num_thresholded_g} eigenvalues in d_g for numerical stability"
                    )

            if classname == 'Conv2d' or classname == 'ConvTranspose2d':
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data

            # Requirement 8.5: Device mismatch handling with clear error messages
            # Ensure Q_a and d_a are on same device as Q_g
            # This is critical for multi-process training where tensors may be on different devices
            if self.Q_a[m].device != self.Q_g[m].device:
                self.device_mismatch_count += 1
                error_msg = DeviceErrorDetector.handle_device_mismatch(
                    f"Q_a[module_{i}]",
                    self.Q_g[m].device,
                    self.Q_a[m].device
                )
                self.logger.warning(error_msg)
                
                # Automatically move tensors to correct device
                self.Q_a[m] = self.Q_a[m].to(self.Q_g[m].device)
                self.d_a[m] = self.d_a[m].to(self.Q_g[m].device)
                
                self.logger.info(
                    f"Moved Q_a and d_a to {self.Q_g[m].device} "
                    f"(total device mismatches: {self.device_mismatch_count})"
                )

            # Compute natural gradient using Kronecker-factored approximation
            # v = Q_g @ (Q_g^T @ grad @ Q_a) / (d_g ⊗ d_a + λ) @ Q_a^T
            try:
                v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
                v2 = v1 / (
                    self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
                v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
                v = v.view(p.grad.data.size())
            except RuntimeError as e:
                self.logger.error(
                    f"KFAC natural gradient computation failed at step {self.steps}, "
                    f"module {i}: {e}. Falling back to SGD."
                )
                self.optim.step()
                self.steps += 1
                return
            
            # Requirement 8.4: Check for NaN/Inf in computed update
            if torch.isnan(v).any() or torch.isinf(v).any():
                self.logger.error(
                    f"NaN or Inf in KFAC update at step {self.steps}, module {i}. "
                    f"Falling back to SGD for this step."
                )
                self.optim.step()
                self.steps += 1
                return
            
            updates[p] = v

        # Compute KL divergence for step size clipping
        vg_sum = 0
        for p in self.model.parameters():
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        # Prevent division by zero or negative values
        vg_sum = max(vg_sum.item(), 1e-10)
        nu = min(1, math.sqrt(self.kl_clip / vg_sum))

        # Apply natural gradient updates
        for p in self.model.parameters():
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        # Requirement 8.4: Final check for NaN/Inf before optimizer step
        has_nan_final, has_inf_final, _ = self.gradient_checker.check_gradients(
            self.model, self.steps
        )
        
        if has_nan_final or has_inf_final:
            self.logger.warning(
                f"NaN or Inf in final gradients at step {self.steps}. "
                f"Applying gradient clipping."
            )
            # Reset gradients to prevent corruption
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optim.step()
        self.steps += 1
        
        # Requirement 8.4: Check parameters after update
        has_nan_params, has_inf_params, _ = self.gradient_checker.check_parameters(
            self.model, self.steps
        )
        
        if has_nan_params or has_inf_params:
            self.logger.critical(
                f"Model parameters corrupted at step {self.steps}! "
                f"Training should be stopped and checkpoint restored."
            )
    
    def get_statistics(self):
        """
        Get optimizer statistics for monitoring.
        
        Returns:
            dict: Statistics including error counts
        """
        grad_stats = self.gradient_checker.get_statistics()
        return {
            'steps': self.steps,
            'nan_inf_count': self.nan_inf_count,
            'device_mismatch_count': self.device_mismatch_count,
            'total_nan_detections': grad_stats['total_nan_detections'],
            'total_inf_detections': grad_stats['total_inf_detections']
        }
