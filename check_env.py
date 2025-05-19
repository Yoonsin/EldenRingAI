from EldenEnv import EldenEnv
from stable_baselines3.common.env_checker import check_env
import gym


logdir = r"C:\GitHub\EldenRingAI\log"
env = EldenEnv(logdir)
#check_env(env)

episodes = 3

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        print('reward', reward)
