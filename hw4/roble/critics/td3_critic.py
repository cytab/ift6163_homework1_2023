from .ddpg_critic import DDPGCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy

from hw4.roble.infrastructure import pytorch_util as ptu
from hw4.roble.policies.MLP_policy import ConcatMLP


class TD3Critic(DDPGCritic):

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        qa_t_values = self.q_net(ob_no, ac_na)
        
        # TODO compute the Q-values from the target network 
        ## Hint: you will need to use the target policy
        ac = self.actor_target(next_ob_no)
        epsilon = torch.randn_like(ac) * self.hparams['alg']['td3_target_policy_noise']
        epsilon = torch.clamp(epsilon, -0.5, 0.5)
        ac_next = ac + epsilon
        action_noise = torch.clamp(ac_next, -self.ac_dim, self.ac_dim)
        qa_tp1_values = self.q_net_target(next_ob_no, action_noise).squeeze(1)

        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma*qa_tp1_values*(1-terminal_n)
        target = target.detach()
        q_t_values = qa_t_values.squeeze(1)
        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        #self.learning_rate_scheduler.step()
        return {'Critic':{
            'Training Loss': ptu.to_numpy(loss),
            'Q Predictions': ptu.to_numpy(q_t_values),
            'Q Targets': ptu.to_numpy(target),
            'Policy Actions': ptu.to_numpy(ac_na),
            'Actor Actions': ptu.to_numpy(self.actor(ob_no)),
            }
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(self.hparams['alg']['polyak_avg']*param.data + (1 - self.hparams['alg']['polyak_avg'])*target_param.data)
            
        for target_param, param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(self.hparams['alg']['polyak_avg']*param.data + (1 - self.hparams['alg']['polyak_avg'])*target_param.data)



