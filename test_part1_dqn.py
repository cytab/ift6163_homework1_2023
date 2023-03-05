import os

os.system("python run_hw3_ql.py env.env_name=MsPacman-v0 env.exp_name=q1 alg.n_iter=1000000")
os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q1_lunar_basic alg.n_iter=350000")

os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q2_dqn1 logging.random_seed=1 alg.n_iter=350000")
os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q2_dqn2 logging.random_seed=2 alg.n_iter=350000")
os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q2_dqn3 logging.random_seed=3 alg.n_iter=350000")


os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn1 logging.random_seed=1 alg.double_q=True alg.n_iter=350000")
os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn2 logging.random_seed=2 alg.double_q=True alg.n_iter=350000")
os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn3 logging.random_seed=3 alg.double_q=True alg.n_iter=350000")


os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q3_hparam1 logging.random_seed=1 alg.n_iter=350000 alg.target_update_freq=1")
os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q3_hparam2 logging.random_seed=2 alg.n_iter=350000 alg.target_update_freq=1000")
os.system("python run_hw3_ql.py env.env_name=LunarLander-v3 env.exp_name=q3_hparam3 logging.random_seed=3 alg.n_iter=350000 alg.target_update_freq=100000")