import gym
from gym import spaces
import numpy as np

class GoalConditionedEnv(gym.Env):

    def __init__(self, env, params):
        self.env = env
        self.params = params
        
        self.goal_distribution = None
        self.new_low_ac = -1
        self.new_high_ac = 1
        if isinstance(self.env, gym.Env):
            if self.params['env']['env_name'] == 'reacher':
                self.new_low_obs = -0.3
                self.new_high_obs = 0.3
                self.shape_ac = (7,)
                self.observation_space = spaces.Box(low=self.new_low_obs, high=self.new_high_obs, shape=(22,))
                self.action_space = spaces.Box(low=self.new_low_ac, high=self.new_high_ac, shape=self.shape_ac)
                
                
            elif self.params['env']['env_name']== 'antmaze':
                self.new_low_obs = -4.0
                self.new_high_obs = 20.0
                self.observation_space = self.env.observation_space()
                self.action_space = self.action_space()
                self.shape_ac = (8,)
                
        
        
        self.goal = np.random.uniform(low=self.new_low_obs, high=self.new_high_obs, size=(2,))
        self.state = None
        self.state_mod = None
    
    def reset(self):
        # Add code to generate a goal from a distribution
        if self.params['env']['env_name'] == 'reacher':
            self.state = self.env.reset()
        elif self.params['env']['env_name']== 'antmaze':
            self.goal = np.random.uniform(low=self.new_low_obs, high=self.new_high_obs, size=(2,))
            self.state = self.env.reset()
        return self.createState() 

    def step(self, action):
        ## Add code to compute a new goal-conditioned reward
        if self.params['env']['env_name'] == 'reacher':
            self.state, new_reward, done, info = self.env.step(action)
        elif self.params['env']['env_name']== 'antmaze':
            self.state, reward, done, info = self.env.step(action)
            new_reward = -np.linalg.norm(self.state[:2] - self.goal)
        
        return self.createState(), new_reward, done, info 
        
    def createState(self):
        ## Add the goal to the state
        return np.append(self.state, self.goal)
        
class GoalConditionedEnvV2(GoalConditionedEnv):

    def __init__(self):
        # TODO
        pass 
    
    def reset(self):
        # Add code to generate a goal from a distribution
        # TODO
        pass 

    def step(self):
        ## Add code to compute a new goal-conditioned reward
        # TODO
        pass 
        