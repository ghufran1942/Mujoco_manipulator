import time
import os

import numpy as np
import gymnasium as gym
from gymnasium.utils.save_video import save_video

import robosuite as suite
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import GymWrapper

from networks import Agent

import imageio

"""
Ensure you have imagemagick installed with
sudo apt-get install imagemagick

Open file in CLI with:
xgd-open <filelname>
"""

if __name__ == "__main__":
    """
    This script demonstrates the usage of the robosuite library with a DDPG agent for reinforcement learning.
    It sets up a robosuite environment (Door), configures a DDPG agent with specified hyperparameters,
    and runs a training loop for a specified number of episodes. The script also includes options
    for loading a pre-trained model and rendering the environment during training.

    The script utilizes the following libraries:
        - time: For pausing execution to control rendering speed.
        - os: For interacting with the operating system (not explicitly used in the provided snippet, but often used for file management).
        - numpy: For numerical operations and handling arrays.
        - gymnasium: For defining the environment interface.
        - torch.utils.tensorboard: For logging training progress and metrics.
        - robosuite: The core library for creating and interacting with robotic environments.
        - robosuite.controllers: For defining robot control schemes.
        - robosuite.wrappers: For wrapping robosuite environments to conform to the Gymnasium interface.

    The script performs the following steps:
        1. Sets up the robosuite environment (Door with a Panda robot).
        2. Configures the robot controller (JOINT_VELOCITY).
        3. Wraps the robosuite environment with a Gymnasium wrapper.
        4. Initializes a DDPG agent with specified hyperparameters (learning rates, batch size, layer sizes, tau).
        5. Optionally loads a pre-trained model.
        6. Runs a training loop for a specified number of episodes:
            - Resets the environment at the beginning of each episode.
            - Chooses an action based on the current observation using the agent's policy.
            - Executes the action in the environment.
            - Renders the environment (if enabled).
            - Calculates the reward and updates the agent's policy and critic.
            - Prints the episode score.
    """

    env_name = "Door"
    robots = ["Panda"]
    controller = "JOINT_VELOCITY"

    # STEP 1: Load part controller
    part_config = load_part_controller_config(default_controller=controller)

    # STEP 2: Refactor into composite-compatible config
    composite_config = refactor_composite_controller_config(part_config, robots[0], ["right"])

    env = suite.make(
        env_name, # name of the environment to load
        robots, # list of robots to use in the environment
        controller_configs=composite_config, # method to control the robot's joints
        has_renderer=True, # whether to render the environment, True -> render, False -> no rendering
        use_camera_obs=False, # whether to use camera observations, True -> use camera observations (for CNN), False -> use joint observations
        horizon=300, # basically max duration for the robot to find the solution
        # renderer="mjviewer",
        render_camera="frontview", # whether to render the environment, True -> render, False -> no rendering
        has_offscreen_renderer=False, # whether to use offscreen rendering, True -> use offscreen rendering, False -> no offscreen rendering
        reward_shaping=True, # False -> only when the robot is successful, True -> partial reward
        control_freq=20 # frequency of control updates
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer_1_size = 256
    layer_2_size = 128
    tau = 0.005

    skip_frames = 1

    print(env.action_space.shape[0])

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=tau, input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], layer_1_size=layer_1_size, layer_2_size=layer_2_size, batch_size=batch_size)

    n_games = 3
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate: {actor_learning_rate}, critic_learning_rate: {critic_learning_rate}, batch_size: {batch_size}, layer_1_size: {layer_1_size}, layer_2_size: {layer_2_size}"

    agent.load_model()

    for i in range(n_games):
        observation, _ = env.reset()
        writer = imageio.get_writer(f"video_test_episode_{i}.mp4", fps=20)
        frames = []
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, validation=True)
            # print(f"Environment Step: {env.step(action)}")
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # env.render()
            if i % skip_frames == 0:
                print(next_observation)
                frame = next_observation
                writer.append_data(frame)
                print("Saving frame #{}".format(i))
            score += reward
            observation = next_observation
            time.sleep(0.03)

        writer.close()
        print(f"Episode {i}: Score {score}")
