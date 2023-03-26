import gym
import numpy as np

class GoalConditionedEnv(gym.Env):

    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.goal_distribution = None
        if isinstance(self.env, gym.Env):
            if self.params['env']['env_name'] == 'reacher':
                self.goal_distribution = np.random.uniform(low=-0.3, high=0.3, size=(2,))
            elif self.params['env']['env_name']== 'antmaze':
                self.goal_distribution = np.random.uniform(low=-4.0, high=20.0, size=(2,))
        self.state = None
        self.state_mod = None
        self.goal = None 
    
    def reset(self):
        # Add code to generate a goal from a distribution
        self.goal = self.goal_distribution.sample()
        self.state = self.env.reset()
        return self.createState() 

    def step(self, action):
        ## Add code to compute a new goal-conditioned reward
        self.state, reward, done, info = self.env.step(action)
        new_reward = -np.linalg.norm(self.state[:2] - self.goal)
        return self.createState(), new_reward, done, info 
        
    def createState(self):
        ## Add the goal to the state
        return np.concatenate((self.state, self.goal))
        
class GoalConditionedEnvV2(gym.Env):

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
        
    def createState(self):
        ## Add the goal to the state
        # TODO
        pass 