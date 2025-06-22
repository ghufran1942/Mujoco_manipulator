from urllib.parse import uses_query
import time
import os

import numpy as np
import gymnasium as gym

from torch.nn.modules import batchnorm
from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

from networks import CriticNetwork, ActorNetwork, Agent
from buffer import ReplayBuffer
from pkg_resources import Environment

def main():
    env_name = "Door"
    robots = ["Panda"]
    controller = "JOINT_VELOCITY"

    # STEP 1: Load part controller
    part_config = load_part_controller_config(default_controller=controller)

    # STEP 2: Refactor into composite-compatible config
    composite_config = refactor_composite_controller_config(part_config, robots[0], ["right"])

    env = suite.make(
        env_name,
        robots,
        controller_configs=composite_config, # method to control the robot's joints
        has_renderer=False,
        use_camera_obs=False, # whether to use camera observations, True -> use camera observations (for CNN), False -> use joint observations
        horizon=300, # basically max duration for the robot to find the solution
        reward_shaping=True, # False -> only when the robot is successful, True -> partial reward
        control_freq=20
    )

    env = GymWrapper(env)

    # critic = CriticNetwork([8],8)
    # actor = ActorNetwork([8],8)

    # replay_buffer = ReplayBuffer(8, [8], 8)
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer_1_size = 256
    layer_2_size = 128
    tau = 0.005

    print(type(env.action_space.shape[0]))

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=tau, input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], layer_1_size=layer_1_size, layer_2_size=layer_2_size, batch_size=batch_size)

    writer = SummaryWriter('logs')
    n_games = 10000
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate: {actor_learning_rate}, critic_learning_rate: {critic_learning_rate}, batch_size: {batch_size}, layer_1_size: {layer_1_size}, layer_2_size: {layer_2_size}"

    agent.load_model()

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            # print(f"Environment Step: {env.step(action)}")
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()

        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if i % 100:
            agent.save_model()

        print(f"Episode {i}: Score {score}")


if __name__ == "__main__":
    main()
