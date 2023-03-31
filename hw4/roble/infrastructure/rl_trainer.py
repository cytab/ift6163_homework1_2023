from collections import OrderedDict
import pickle
import os
import sys
import time
import gym
from gym import wrappers
import numpy as np
import torch

sys.path.append('/home/cyrille/Desktop/IFT6163/ift6163_homework1_2023')

from hw4.roble.infrastructure import pytorch_util as ptu
from hw4.roble.infrastructure import utils
from hw4.roble.infrastructure.logger import Logger
from hw4.roble.agents.dqn_agent import DQNAgent
from hw4.roble.agents.ddpg_agent import DDPGAgent
from hw4.roble.agents.td3_agent import TD3Agent
from hw4.roble.agents.sac_agent import SACAgent
from hw4.roble.agents.pg_agent import PGAgent

from hw4.roble.infrastructure.dqn_utils import (
        get_wrapper_by_name
)
from hw4.roble.envs.ant.create_maze_env import create_maze_env
from hw4.roble.envs.reacher.reacher_env import create_reacher_env
from hw4.roble.infrastructure.gclr_wrapper import GoalConditionedEnv, GoalConditionedEnvV2
from hw4.roble.infrastructure.hrl_wrapper import HRLWrapper
# how many rollouts to save as videos
MAX_NVIDEO = 1
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class RL_Trainer(object):

    def __init__(self, params, agent_class = None):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logging']['logdir'])

        # Set random seeds
        seed = self.params['logging']['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['alg']['no_gpu'],
            gpu_id=self.params['alg']['which_gpu']
        )

        #############
        ## ENV
        #############
        # Make the gym environment
        if self.params['env']['env_name'] == 'antmaze':
            self.env = create_maze_env('AntMaze')
        elif self.params['env']['env_name'] == 'reacher':
            self.env = create_reacher_env()
     
        
    
            
            
        # Call your goal conditionned wrapper here (You can modify arguments depending on your implementation)
        if self.params['env']['task_name'] == 'gcrl':
            self.env = GoalConditionedEnv(self.env, self.params)     
        elif self.params['env']['task_name'] == 'gcrl_v2':
            self.env = GoalConditionedEnvV2(self.env, self.params)
        elif self.params['env']['task_name'] == 'hrl':
            self.env = HRLWrapper(self.env, self.params)



            
            
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.Monitor(
                self.env,
                os.path.join(self.params['logging']['logdir'], "gym"),
                force=True,
                video_callable=(None if self.params['logging']['video_log_freq'] > 0 else False),
            )
            self.env = params['env_wrappers'](self.env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        if 'non_atari_colab_env' in self.params and self.params['logging']['video_log_freq'] > 0:
            self.env = wrappers.Monitor(
                self.env,
                os.path.join(self.params['logging']['logdir'], "gym"),
                force=True,
                video_callable=(None if self.params['logging']['video_log_freq'] > 0 else False),
            )
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        
        self.env.seed(seed)

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env']['env_name']=='obstacles-roble-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['env']['max_episode_length'] = self.params['env']['max_episode_length'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['env']['max_episode_length']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['alg']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['alg']['ac_dim'] = ac_dim
        self.params['alg']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        #elif 'video.frames_per_second' in self.env.env.metadata.keys():
            #self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10


        #############
        ## AGENT
        #############

        # agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params)

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1000 if isinstance(self.agent, DQNAgent) else 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['logging']['video_log_freq'] == 0 and self.params['logging']['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['logging']['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['logging']['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, DQNAgent) or isinstance(self.agent, DDPGAgent):
                # only perform an env step and add to replay buffer for DQN and DDPG
                self.agent.step_env()
                envsteps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                use_batchsize = self.params['alg']['batch_size']
                if itr==0:
                    use_batchsize = self.params['alg']['batch_size_initial']
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, initial_expertdata, collect_policy, use_batchsize)
                )

            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()
    

            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                if isinstance(self.agent, DQNAgent):
                    self.perform_dqn_logging(itr, all_logs)
                
                elif isinstance(self.agent, DDPGAgent):
                    self.perform_ddpg_logging(itr,all_logs)
                    
                else:
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['logging']['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logging']['logdir'], itr))
            if  self.params['alg']['on_policy']:       
                self.agent.clear_mem()

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # TODO: get this from hw1 or hw2
        if itr == 0 and initial_expertdata != None:
            with open(initial_expertdata, "rb") as file :
                loaded_paths = pickle.load(file)
                return loaded_paths, 0, None
        else: 
            # TODO collect `batch_size` samples to be used for training
            # HINT1: use sample_trajectories from utils
            # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
            print("\nCollecting data to be used for training...")
            paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, num_transitions_to_sample, self.params['env']['max_episode_length'])
            
        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN

        train_video_paths = None
        if self.logvideo:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
    
        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['alg']['num_agent_train_steps_per_iter']):
            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['alg']['train_batch_size'])

            # TODO use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################
    def perform_dqn_logging(self, itr, all_logs):
        last_log = all_logs[-1]

        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        self.logger.log_file(itr, logs)
        print('Done DQN logging...\n\n')

        self.logger.flush()
        
    
    def perform_ddpg_logging(self, itr, all_logs):
        logs = OrderedDict()
        logs["Train_EnvstepsSoFar"] = self.agent.t
        
        n = 25
        if len(self.agent.rewards) > 0:
            self.mean_episode_reward = np.mean(np.array(self.agent.rewards)[-n:])
            
            logs["Train_AverageReward"] = self.mean_episode_reward
            logs["Train_CurrentReward"] = self.agent.rewards[-1]
            
        if len(self.agent.rewards) > n:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
            
            logs["Train_BestReturn"] = self.best_mean_episode_reward
            
        if len(self.agent.rewards) > 10 * n:   
            self.agent.rewards = self.agent.rewards[n:]
            
        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            logs["TimeSinceStart"] = time_since_start
        
        Q_predictions = []
        Q_targets = []
        policy_actions_mean = []
        policy_actions_std = []
        actor_actions_mean = []
        critic_loss = []
        actor_loss = []
        print_all_logs = True
        for log in all_logs:
            if len(log) > 0:
                print_all_logs = True
                #print(Q_predictions)
                Q_predictions.append(np.mean(log["Critic"]["Q Predictions"]))
                Q_targets.append(np.mean((log["Critic"]["Q Targets"])))
                policy_actions_mean.append(np.mean((log["Critic"]["Policy Actions"])))
                policy_actions_std.append(np.std((log["Critic"]["Policy Actions"])))
                actor_actions_mean.append(np.mean((log["Critic"]["Actor Actions"])))
                critic_loss.append(log["Critic"]["Training Loss"])
                
                if "Actor" in log.keys():
                    actor_loss.append(log["Actor"])
                
        if print_all_logs:
            logs["Q_Predictions"] = np.mean(np.array(Q_predictions))
            logs["Q_Targets"] = np.mean(np.array(Q_targets))
            logs["Policy_Actions_Mean"] = np.mean(np.array(policy_actions_mean))
            logs["Policy_Actions_Std"] = np.mean(np.array(policy_actions_std))
            logs["Actor_Actions"] = np.mean(np.array(actor_actions_mean))
            logs["Critic_Loss"] = np.mean(np.array(critic_loss))
            
            if len(actor_loss) > 0:
                logs["Actor_Loss"] = np.mean(np.array(actor_loss))
            
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        self.logger.log_file(itr, logs)
        print('Done DDPG logging...\n\n')
        #logs.update(last_log)
        self.logger.flush()

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):
        
        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['alg']['eval_batch_size'], self.params['env']['max_episode_length'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo:
            if train_video_paths is not None:
                #save train/eval videos
                print('\nSaving train rollouts as videos...')
                self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            else:
                print('\nCollecting video rollouts eval')
                eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
                print('\nSaving eval rollouts as videos...')
                self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].mean() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = all_logs[-1]
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            self.logger.log_file(itr, logs)
            print('Done logging...\n\n')

            self.logger.flush()