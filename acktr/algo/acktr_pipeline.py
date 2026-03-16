import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from acktr.algo.kfac import KFACOptimizer
import sys

class ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 invaild_coef,
                 lr = None,
                 eps = None,
                 alpha = None,
                 max_grad_norm = None,
                 acktr = False,
                 args = None):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.invaild_coef = invaild_coef
        self.max_grad_norm = max_grad_norm

        self.loss_func = nn.MSELoss(reduce=False, size_average=True)
        self.entropy_coef = entropy_coef
        self.args = args

        if acktr:
            # Use paper-specified KFAC parameters from args if available
            # Requirements 7.4: stat_decay=0.99, kl_clip=0.001, damping=1e-2
            stat_decay = args.stat_decay if args and hasattr(args, 'stat_decay') else 0.99
            kl_clip = args.kl_clip if args and hasattr(args, 'kl_clip') else 0.001
            damping = args.damping if args and hasattr(args, 'damping') else 1e-2
            
            self.optimizer = KFACOptimizer(
                actor_critic,
                stat_decay=stat_decay,
                kl_clip=kl_clip,
                damping=damping
            )
        else:
            self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)


    def update(self, rollouts):
        # check_nan(self.actor_critic, 1)
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        mask_size = rollouts.location_masks.size()[-1]

        # Compute expected batch size for multi-process training
        # Requirements 1.3, 1.4: Batch size should equal num_steps × num_processes
        expected_batch_size = num_steps * num_processes

        # Reshape from (num_steps, num_processes, ...) to (num_steps * num_processes, ...)
        # This aggregates experiences from all parallel environments for batched gradient computation
        # Requirements 1.3, 1.4, 2.1: Ensure all processes contribute to gradient updates
        obs_batch = rollouts.obs[:-1].view(-1, *obs_shape)
        actions_batch = rollouts.actions.view(-1, action_shape)
        masks_batch = rollouts.masks[:-1].view(-1, 1)
        location_masks_batch = rollouts.location_masks[:-1].view(-1, mask_size)
        
        # Verify batch size matches expected (num_steps * num_processes)
        assert obs_batch.size(0) == expected_batch_size, \
            f"Batch size mismatch: got {obs_batch.size(0)}, expected {expected_batch_size} " \
            f"(num_steps={num_steps} × num_processes={num_processes})"

        values, action_log_probs, dist_entropy, _, bad_prob, pred_mask = self.actor_critic.evaluate_actions(
            obs_batch,
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            masks_batch,
            actions_batch,
            location_masks_batch)

        # Verify output batch size matches input
        assert values.size(0) == expected_batch_size, \
            f"Values batch size mismatch: got {values.size(0)}, expected {expected_batch_size}"
        assert action_log_probs.size(0) == expected_batch_size, \
            f"Action log probs batch size mismatch: got {action_log_probs.size(0)}, expected {expected_batch_size}"

        # Reshape back to (num_steps, num_processes, 1) for loss computation
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        mask_len = self.args.container_size[0]*self.args.container_size[1]
        mask_len = mask_len * (1+ self.args.enable_rotation)
        pred_mask = pred_mask.reshape((num_steps,num_processes,mask_len))

        mask_truth = rollouts.location_masks[0:num_steps] 
        graph_loss = self.loss_func(pred_mask, mask_truth).mean()
        dist_entropy = dist_entropy.mean()
        prob_loss = bad_prob.mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            # Requirements 2.1: Fisher information matrices accumulate statistics across all processes
            # The .mean() operation averages over all (num_steps * num_processes) samples
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean() # detach

            # Fisher loss includes contributions from all parallel processes
            fisher_loss = pg_fisher_loss + vf_fisher_loss + graph_loss * 1e-8
            # fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        force = 0.5 * 10
        self.optimizer.zero_grad()
        # Requirements 1.4, 2.1: Compute loss using batched gradients from all processes
        # All loss components (value_loss, action_loss, etc.) are computed over
        # (num_steps * num_processes) samples, ensuring gradient accumulation across all processes
        loss = value_loss * self.value_loss_coef
        loss += action_loss
        loss += prob_loss * self.invaild_coef
        loss -= dist_entropy * self.entropy_coef
        loss += force * graph_loss
        loss.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # return value_loss.item(), action_loss.item(), dist_entropy.item(), prob_loss.item()
        return value_loss.item(), action_loss.item(), dist_entropy.item(), prob_loss.item(), graph_loss.item()

def check_nan(model,index):
    for p in model.parameters():
        if np.isnan(p.grad.data.mean().item()):
            print('index '+ str(index) +' happened an error!')