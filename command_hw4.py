import os

os.system("python run_hw4_gc.py env.env_name=reacher env.exp_name=q2_reacher_normal_seed1 alg.goal_dist=normal logging.random_seed=1")
os.system("python run_hw4_gc.py env.env_name=reacher env.exp_name=q2_reacher_normal_seed2 alg.goal_dist=normal logging.random_seed=2")

os.system("python run_hw4_gc.py env.env_name=reacher env.exp_name=q2_reacher_uniform_seed1 alg.goal_dist='uniform' logging.random_seed=1")
os.system("python run_hw4_gc.py env.env_name=reacher env.exp_name=q2_reacher_uniform_seed2 alg.goal_dist='uniform' logging.random_seed=2")

os.system("python run_hw4_gc.py env.env_name=reacher env.exp_name=q3_reacher_normal_relative alg.goal_dist='normal' alg.relative_goal=True")
os.system("python run_hw4_gc.py env.env_name=reacher env.exp_name=q3_reacher_uniform_relative alg.goal_dist='uniform' alg.relative_goal=True")

os.system("python run_hw4_gc.py env.env_name=reacher env.exp_name=q4_reacher_normal_relative alg.goal_dist='normal' alg.relative_goal=True alg.goal_frequency=5")

os.system("python run_hw4_gc.py env.env_name=reacher env.exp_name=q4_reacher_normal_relative alg.goal_dist='normal' alg.relative_goal=True alg.goal_frequency=10")


