import gym
from gym import spaces
import numpy as np
from hw4.roble.agents.pg_agent import PGAgent
from hw4.roble.policies.MLP_policy import MLPPolicyPG

class HRLWrapper(gym.Env):

    def __init__(self, env, params):
        self.env = env
        self.params = params
        
        self.goal_distribution = True if self.params['alg']['goal_dist']=='normal' else False
        self.new_low_ac = -1
        self.new_high_ac = 1
        if isinstance(self.env, gym.Env):
            if self.params['env']['env_name'] == 'reacher':
                self.new_low_obs = -0.3
                self.new_high_obs = 0.3
                self.shape_ac = (7,)
                self.observation_space = spaces.Box(low=self.new_low_obs, high=self.new_high_obs, shape=(20,))
                #self.action_space = spaces.Box(low=self.new_low_ac, high=self.new_high_ac, shape=self.shape_ac)
                self.action_space = self.env.action_space
                #goal_space = gym.spaces.Box(low=self.goal_distribution.min(), high=self.goal_distribution.max(), shape=self.goal_distribution.shape, dtype=np.float32)
                #self.observation_space = gym.spaces.Tuple((self.env.observation_space, goal_space))
                
                
            elif self.params['env']['env_name']== 'antmaze':
                self.new_low_obs = -4.0
                self.new_high_obs = 20.0
                self.observation_space = self.env.observation_space()
                self.action_space = self.action_space()
                self.shape_ac = (8,)
                  
        self.goal = None
        self.state = None
        self.stepK = 0
        self.reset()


        filepath = "/home/bassem/RobotLearning/ift6163_homeworks_2023/hw4/roble/policies/Trained_policies/{}_policy.pth".format(self.params['alg']['loaded_policy_name'])
        self.agent_params = self.params['alg']

        self.LLC_agent = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline'],
            learn_policy_std=self.agent_params['learn_policy_std'],
        )

        self.LLC_agent.load(filepath=filepath) #add path

        # to use the low level policy

        
    
    def reset(self):
        # Add code to generate a goal from a distribution
        self.stepK = 0
        if self.params['env']['env_name'] == 'reacher':
            self.state = self.env.reset()
            self.goal = self.sample_goal(self.goal_distribution)
            self.state[-3:] = self.goal - self.params['alg']['relative_goal']*self.state[-6:-3]
            return self.state
        elif self.params['env']['env_name']== 'antmaze':
            self.state = self.env.reset()
            self.goal = self.sample_goal(self.goal_distribution)
            return self.createState() 
    
    def render(self, mode):
        return self.env.render(mode)

    def step(self, action):
        ## Add code to compute a new goal-conditioned reward

        self.sub_goal = action # The high level policy action \pi(g|s,\theta^{hi}) is the low level goal.
        for _ in range(self.agent_params['goal_frequency']):
            ## Get the action to apply in the environment
            ## HINT you need to use \pi(a|s,g,\theta^{low})
            ## Step the environment
            self.stepK +=1
            
            if self.params['env']['env_name'] == 'reacher' :
                self.state[-3:] = self.sub_goal - self.params['alg']['relative_goal']*self.state[-6:-3]
                action_LLC = self.LLC_agent(self.state)
                self.state, new_reward, done, info = self.env.step(action_LLC)
                return self.state, new_reward, done, info 
            
            elif self.params['env']['env_name']== 'antmaze':
                action_LLC = self.LLC_agent(self.createState_LLC)
                self.state, reward, done, info = self.env.step(action_LLC)
                new_reward = -np.linalg.norm(self.state[:2] - self.goal)
                return self.createState(), new_reward, done, info 
        
    def createState(self):
        ## Add the goal to the state
        return np.append(self.state, self.goal - self.params['alg']['relative_goal']*self.state)

    def createState_LLC(self):
        ## Add the goal to the state
        return np.append(self.state, self.sub_goal - self.params['alg']['relative_goal']*self.state)