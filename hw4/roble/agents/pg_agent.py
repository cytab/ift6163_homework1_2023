import numpy as np

from .base_agent import BaseAgent
from hw4.roble.policies.MLP_policy import MLPPolicyPG
from hw4.roble.infrastructure.replay_buffer import ReplayBuffer

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['alg']['discount']
        self.standardize_advantages = self.agent_params['alg']['standardize_advantages']
        self.nn_baseline = self.agent_params['alg']['nn_baseline']
        self.reward_to_go = self.agent_params['alg']['reward_to_go']
        self.gae_lambda = self.agent_params['alg']['gae_lambda']

        if self.gae_lambda == 'None':
            self.gae_lambda = None

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['alg']['ac_dim'],
            self.agent_params['alg']['ob_dim'],
            self.agent_params['alg']['n_layers'],
            self.agent_params['alg']['size'],
            discrete=self.agent_params['alg']['discrete'],
            learning_rate=self.agent_params['alg']['learning_rate'],
            nn_baseline=self.agent_params['alg']['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)
        train_log = self.actor.update(observations, actions, advantages=advantages, q_values=q_values)

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        q_values = []

        if not self.reward_to_go:
            for rollout in rewards_list:
                q_values += list(self._discounted_return(rollout))

        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            for rollout in rewards_list:
                q_values += list(self._discounted_cumsum(rollout))

        return np.array(q_values)

    def estimate_advantage(self, obs, rews_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            ## Values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values

            values_normalized = (values_unnormalized - values_unnormalized.mean()) / (values_unnormalized.std() + 1e-8)
            values = values_normalized * np.std(q_values) + np.mean(q_values)

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ended = 1 - terminals[i]
                    delta_t = rews[i] + (self.gamma * values[i+1] * ended) - values[i]
                    advantages[i] = delta_t + (self.gae_lambda * self.gamma * advantages[i+1] * ended) 

                # remove dummy advantage
                advantages = advantages[:-1]
            else:
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    def save(self, path):
        return self.actor.save(path)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        indices = np.arange(len(rewards))
        gamma_indices = np.power(self.gamma,indices)
        discount_return = np.sum(gamma_indices * rewards)
        list_of_discounted_returns = np.array([discount_return] * len(rewards))

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        indices = np.arange(len(rewards))
        indices = np.arange(len(rewards))
        indices_matrix = np.tile(indices, (len(indices), 1))
        indices_matrix = (indices_matrix.T - indices).T
        indices_matrix = np.where(indices_matrix <0, 0, indices_matrix)

        gamma_matrix = np.triu(self.gamma ** indices_matrix)

        list_of_discounted_cumsums = np.sum(gamma_matrix * rewards,axis=1)

        return list_of_discounted_cumsums
