import os

os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up1_criticlr1e-1 alg.n_iter=20000 alg.critic_learning_rate=1e-1 env.atari=False alg.rl_alg='ddpg'")
os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up1_criticlr1e-4 alg.n_iter=20000 alg.critic_learning_rate=1e-4 env.atari=False alg.rl_alg='ddpg'")
os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up1_criticlr1e-5 alg.n_iter=20000 alg.critic_learning_rate=1e-5 env.atari=False alg.rl_alg='ddpg'")

os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up2_lr1e-3 alg.n_iter=20000 alg.num_agent_train_steps_per_iter=2 env.atari=False alg.rl_alg='ddpg'")
os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up10_lr1e-3 alg.n_iter=20000 alg.num_agent_train_steps_per_iter=10 env.atari=False alg.rl_alg='ddpg'")
os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up100_lr1e-3 alg.n_iter=20000 alg.num_agent_train_steps_per_iter.=100 env.atari=False alg.rl_alg='ddpg'")

os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up1_actorlr1e-1 alg.n_iter=20000 alg.learning_rate=1e-1 env.atari=False alg.rl_alg='ddpg'")
os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up1_actorlr1e-4 alg.n_iter=20000 alg.learning_rate=1e-4 env.atari=False alg.rl_alg='ddpg'")
os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up1_actorlr1e-5 alg.n_iter=20000 alg.learning_rate=1e-5 env.atari=False alg.rl_alg='ddpg'")

