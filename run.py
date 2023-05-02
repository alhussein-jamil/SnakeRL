
from snake_env import SnakeEnv
import ray
import imageio
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
import matplotlib.pyplot as plt
import mediapy as media
from datetime import datetime
from pathlib import Path
ray.init()
print("Ray initialized")

def env_creator(env_config):
    return SnakeEnv(env_config)

register_env("snake-v0", env_creator)

#read the yaml as dictionary 
import yaml
with open("SnakeConfig.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        if config is None:
            raise Exception("Invalid YAML file")
    except yaml.YAMLError as exc:
        print(exc)

print("Config loaded")
fps = 20



trainer = PPOTrainer(config=config, env="snake-v0")

snakie = SnakeEnv({"render_mode": "rgb_array"})

SIMPATH = "./simulations/"+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/"
Path(SIMPATH).mkdir(parents=True, exist_ok=True)

for i in range(1000):
    result = trainer.train()
    print("iteration: ", i, "episode_reward_mean: ", result["episode_reward_mean"], "episode_reward_min: ", result["episode_reward_min"], "episode_reward_max: ", result["episode_reward_max"])

    if i % 10 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

    if(i % 10 == 0):
    
        snakie.reset()
        done = False
        if(snakie.render_mode == "human"):
            snakie.render()
            #wait a bit 
            plt.pause(1/fps)
        
        else: 
            frames = []
            while not done and len(frames) < 300:
                action = trainer.compute_single_action(snakie._get_obs())
                obs, reward, done, _ , _= snakie.step(action)
                frames.append(snakie.render())


            media.write_video(SIMPATH + "snake " + str(i) + ".mp4", frames, fps=fps)
        