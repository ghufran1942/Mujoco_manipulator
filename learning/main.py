import time
import os

import numpy as np
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

from networks import Agent

def main():
    """
    This is the main function that sets up the Robosuite environment,
    defines the agent, and runs the training loop.

    You can modify the following variables to experiment with different settings:
        - env_name: The name of the Robosuite environment to use.
        - robots: The robot(s) to use in the environment.
        - controller: The type of controller to use for the robot(s).
        - actor_learning_rate: The learning rate for the actor network.
        - critic_learning_rate: The learning rate for the critic network.
        - batch_size: The batch size for training.
        - layer_1_size: The size of the first hidden layer in the actor and critic networks.
        - layer_2_size: The size of the second hidden layer in the actor and critic networks.
        - tau: The soft update coefficient for the target networks.
        - n_games: The number of training episodes to run.
    """
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

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer_1_size = 256
    layer_2_size = 128
    tau = 0.005

    print(env.action_space.shape[0])

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=tau, input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], layer_1_size=layer_1_size, layer_2_size=layer_2_size, batch_size=batch_size)

    writer = SummaryWriter('logs')
    n_games = 5000
    # best_score = 0
    episode_identifier = f"0 - actor_learning_rate: {actor_learning_rate}, critic_learning_rate: {critic_learning_rate}, batch_size: {batch_size}, layer_1_size: {layer_1_size}, layer_2_size: {layer_2_size}"

    # agent.load_model()

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
            observation = next_observation

        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if (i % 10 == 0):
            agent.save_model()
        print(f"Episode {i}: Score {score}")


if __name__ == "__main__":
    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")
    main()
