import time
import os

import numpy as np
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import GymWrapper

if __name__ == "__main__":
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
        # controller_configs=composite_config, # method to control the robot's joints
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
