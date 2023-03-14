import os
os.system("python run_hw3_ql.py env.env_name=InvertedPendulum-v2 alg.target_update_freq=1 env.exp_name=q4_ddpg_up1_criticlr1e-3 alg.n_iter=350000 alg.learning_rate=1e-3 env.atari=False alg.rl_alg='sac' alg.discrete=False")
