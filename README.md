# Mujoco Manipulator

This repository provides a minimal example of training a robotic manipulator in the [Robosuite](https://robosuite.ai/) simulator. The agent uses a TD3 algorithm implemented in PyTorch.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r learning/requirements.txt
```

## Training

To start training run:

```bash
python learning/main.py 
```

Training logs are written to the `logs/` directory and model checkpoints are stored in `tmp/td3/`.

## Evaluation

After training, visualize the learned policy by running:

```bash
python learning/test.py
```

If you are on **macOS**, use the MuJoCo provided `mjpython` interpreter when running any scripts or tests. For example:

```bash
mjpython learning/test.py # To be used only with has_renderer=True & has_offscreen_renderer=False
```

For test script with video recording, use `python` instead of `mjpython`.

This will load the latest checkpoints and render the environment while the agent interacts with it.

## Visualizing with TensorBoard

You can monitor training metrics using [TensorBoard](https://www.tensorflow.org/tensorboard). Run the following command and open the displayed URL in your browser:

```bash
tensorboard --logdir="logs" --port 6006
```
