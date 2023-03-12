from .ddpg_critic import DDPGCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy

from hw3.roble.infrastructure import pytorch_util as ptu
from hw3.roble.policies.MLP_policy import ConcatMLP


class SACCritic(DDPGCritic):

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
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        qa_t_values = self.q_net(ob_no, ac_na)
        
        # TODO compute the Q-values from the target network 
        ## Hint: you will need to use the target policy
        action, log_action = self.actor_target.get_action(next_ob_no)
        qa_tp1_values = self.q_net_target(next_ob_no, action) 

        # TODO add the entropy term to the Q-values
        ## Hint: you will need the use the lob_prob function from the distribution of the actor policy
        ## Hint: use the self.hparams['alg']['sac_entropy_coeff'] value for the entropy term
        
        qa_tp1_values_reg = qa_tp1_values - self.hparams['alg']['sac_entropy_coeff']*log_action 

        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma*qa_tp1_values*(1-terminal_n)
        target = target.detach()

        assert qa_t_values.shape == target.shape
        loss = self.loss(qa_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            ## Perform Polyak averaging
            y = target_param.data.copy(self.hparams['alg']['polyak_avg']*param.data + (1 - self.hparams['alg']['polyak_avg'])*target_param.data)
        for target_param, param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            ## Perform Polyak averaging for the target policy
            y = target_param.data.copy(self.hparams['alg']['polyak_avg']*param.data + (1 - self.hparams['alg']['polyak_avg'])*target_param.data)

