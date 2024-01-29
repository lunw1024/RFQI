from gymnasium.core import ObservationWrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gymnasium as gym
from gymnasium import spaces
from sustaingym.envs.building import BuildingEnv, ParameterGenerator
from tqdm.auto import tqdm, trange

if __name__ == "__main__":
    params = ParameterGenerator(
        building='OfficeSmall', weather='Hot_Dry', location='Tucson', time_res=300)
    env = BuildingEnv(params)

    ZONE_NAMES = ['South', 'East', 'North', 'West', 'Core', 'Plenum', 'Outside']
    num_steps = 288  # 5 min intervals for 1 day

    print('Size of State Space:', env.observation_space.shape)
    print('Size of Action Space:', env.action_space.shape)
    print('Sample State:', env.observation_space.sample())
    print('Sample Action:', env.action_space.sample())
    
    model = PPO(MlpPolicy, env, verbose=1)


    reward_history = []
    for i in trange(100):
        model.learn(10000)
        reward = 0
        vec_env = model.get_env()
        assert vec_env != None
        obs = vec_env.reset()
        # evaluate
        for j in range(24):
            action, states = model.predict(obs)
            obs, rewards, dones, infos = vec_env.step(action)
            reward += rewards
        print(f"Avg reward = {reward / 24}")
        reward_history.append(reward / 24)
    model.save('artifacts/PPO.zip')
