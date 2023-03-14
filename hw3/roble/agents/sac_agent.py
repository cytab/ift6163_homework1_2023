import numpy as np
import copy

from hw3.roble.infrastructure.replay_buffer import ReplayBuffer
from hw3.roble.policies.MLP_policy import MLPPolicyStochastic
from hw3.roble.critics.sac_critic import SACCritic
from hw3.roble.agents.ddpg_agent import DDPGAgent

class SACAgent(DDPGAgent):
    def __init__(self, env, agent_params):

        super().__init__(env, agent_params)
        
        self.actor = MLPPolicyStochastic(
            self.agent_params['alg']['sac_entropy_coeff'],
            self.agent_params['alg']['ac_dim'],
            self.agent_params['alg']['ob_dim'],
            self.agent_params['alg']['n_layers'],
            self.agent_params['alg']['size'],
            discrete=self.agent_params['alg']['discrete'],
            learning_rate=self.agent_params['alg']['learning_rate'],
            nn_baseline=False,
        )

        self.q_fun = SACCritic(self.actor, 
                               agent_params, 
                               self.optimizer_spec)
        
    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # TODO: Take the code from DDPG Agent and make sure the remove the exploration noise
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        # TODO add noise to the deterministic policy
        perform_random_action = True if (self.t < self.learning_starts) else False
        # HINT: take random action 
        a, log = self.actor.get_action(self.replay_buffer.encode_recent_observation())
        action = self.env.action_space.sample() if perform_random_action else np.clip((a + 0.1*np.random.normal(0, 1)), -self.num_actions, self.num_actions)
        
        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        self.last_obs, reward, done, info = self.env.step(action)
        self.cumulated_rewards += reward
        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()
            self.rewards.append(self.cumulated_rewards)
            self.cumulated_rewards = 0
        return