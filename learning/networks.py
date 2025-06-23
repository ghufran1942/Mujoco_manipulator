import os

import numpy as np

import torch as T
from torch._C import parse_schema
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions.normal as normal

from buffer import ReplayBuffer

class CriticNetwork(nn.Module): # initialize the Critic Network
    """
    Critic network for the TD3 algorithm. It takes the state and action as input and outputs a Q-value.
    It has two fully connected layers with ReLU activation functions, followed by an output layer.

    The critic network is used to evaluate the value of a given state-action pair.
    It is trained to minimize the mean squared error between its predicted Q-values and the target Q-values.

    Attributes:
        input_dims (tuple): The dimensions of the input state.
        n_actions (int): The number of possible actions.
        fc1_dims (int): The number of units in the first fully connected layer.
        fc2_dims (int): The number of units in the second fully connected layer.
        name (str): The name of the network.
        checkpoint_dir (str): The directory to save the checkpoints.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay parameter for regularization.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        q1 (nn.Linear): The output layer for Q-value.
        optimizer (optim.AdamW): The optimizer for the network.
        device (torch.device): The device to use for training.
        checkpoint_file (str): The path to the checkpoint file.
    """
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128, name="critic", checkpoint_dir="tmp/td3", learning_rate=10e-3, weight_decay=0.005):
        """
        Initializes the CriticNetwork.

        Args:
            input_dims (tuple): The dimensions of the input state.
            n_actions (int): The number of possible actions.
            fc1_dims (int): The number of units in the first fully connected layer. Defaults to 256.
            fc2_dims (int): The number of units in the second fully connected layer. Defaults to 128.
            name (str): The name of the network. Defaults to "critic".
            checkpoint_dir (str): The directory to save the checkpoints. Defaults to "tmp/td3".
            learning_rate (float): The learning rate for the optimizer. Defaults to 10e-3.
            weight_decay (float): The weight decay parameter for regularization. Defaults to 0.005.
        """
        super(CriticNetwork, self).__init__() # initialize the parent class

        self.input_dims = input_dims # input dimensions
        self.n_actions = n_actions # number of actions
        self.fc1_dims = fc1_dims # first fully connected layer dimensions
        self.fc2_dims = fc2_dims # second fully connected layer dimensions
        self.name = name # name of the network
        self.checkpoint_dir = checkpoint_dir # directory to save checkpoints
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "_td3")

        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims) # first fully connected layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) # second fully connected layer
        self.q1 = nn.Linear(self.fc2_dims, 1) # output layer for Q-value

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay) # optimizer for the network

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # device to use for training, for mac it is 'cpu' and for laptop it should be cuda:0

        print(f"{self.name} network initialized on device: {self.device}") # print the device the network is initialized on

        self.to(self.device) # move the network to the device

    def forward(self, state, action): # Forward pass for the network
        """
        Forward pass of the critic network.

        Args:
            state (torch.Tensor): The state tensor.
            action (torch.Tensor): The action tensor.

        Returns:
            torch.Tensor: The Q-value for the given state-action pair.
        """
        action_value = self.fc1(T.cat([state, action], dim=1)) # Concatenate state and action tensors along the first dimension
        action_value = F.relu(action_value) # Apply ReLU activation function to the concatenated tensor
        action_value = self.fc2(action_value) # Apply second fully connected layer
        action_value = F.relu(action_value) # Apply ReLU activation function to the output of the second fully connected layer

        q1 = self.q1(action_value) # Apply output layer to the output of the second fully connected layer

        return q1

    def save_checkpoint(self):
        """
        Saves the checkpoint of the network.
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Loads the checkpoint from the specified file.

        Returns:
            bool: True if the checkpoint was successfully loaded, False otherwise.
        """
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
            return True
        else:
            print(f"Checkpoint not found: {self.checkpoint_file}")
            return False


class ActorNetwork(nn.Module):
    """
    A neural network for the actor in a reinforcement learning agent.

    The actor network takes a state as input and outputs an action.
    It consists of two fully connected layers with ReLU activation functions,
    followed by a fully connected layer with a Tanh activation function to
    constrain the output to the action space.

    Attributes:
        input_dims (tuple): The dimensions of the input state.
        fc1_dims (int): The number of neurons in the first fully connected layer.
        fc2_dims (int): The number of neurons in the second fully connected layer.
        learning_rate (float): The learning rate for the optimizer.
        n_actions (int): The number of actions the agent can take.
        name (str): The name of the network.
        checkpoint_dir (str): The directory to save the checkpoint.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        action (nn.Linear): The output layer.
        optimizer (optim.Adam): The optimizer for the network.
        device (torch.device): The device to use for training.
        checkpoint_file (str): The path to the checkpoint file.
    """
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, learning_rate=10e-3, n_actions=2, name='actor', checkpoint_dir='tmp/td3'):
        """
        Initializes the ActorNetwork.

        Args:
            input_dims (tuple): The dimensions of the input state.
            fc1_dims (int): The number of neurons in the first fully connected layer. Defaults to 256.
            fc2_dims (int): The number of neurons in the second fully connected layer. Defaults to 128.
            learning_rate (float): The learning rate for the optimizer. Defaults to 10e-3.
            n_actions (int): The number of actions the agent can take. Defaults to 2.
            name (str): The name of the network. Defaults to 'actor'.
            checkpoint_dir (str): The directory to save the checkpoint. Defaults to 'tmp/td3'.
        """
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims # input dimensions
        self.n_actions = n_actions # number of actions
        self.fc1_dims = fc1_dims # first fully connected layer dimensions
        self.fc2_dims = fc2_dims # second fully connected layer dimensions
        self.name = name # name of the network
        self.checkpoint_dir = checkpoint_dir # directory to save checkpoints
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "_td3")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.action = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # device to use for training, for mac it is 'cpu' and for laptop it should be cuda:0

        print(f"{self.name} network initialized on device: {self.device}") # print the device the network is initialized on

        self.to(self.device) # move the network to the device


    def forward(self, state): # Forward pass for the network
        """
        Forward pass for the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output action tensor.
        """
        x = self.fc1(state) # Concatenate state and action tensors along the first dimension
        x = F.relu(x) # Apply ReLU activation function to the concatenated tensor
        x = self.fc2(x) # Apply second fully connected layer
        x = F.relu(x) # Apply ReLU activation function to the output of the second fully connected layer

        x = T.tanh(self.action(x))

        return x

    def save_checkpoint(self):
        """
        Saves the checkpoint of the network.
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Loads the checkpoint from the specified file.

        Returns:
            bool: True if the checkpoint was successfully loaded, False otherwise.
        """
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
            return True
        else:
            print(f"Checkpoint not found: {self.checkpoint_file}")
            return False


class Agent():
    """
    A class representing an agent that uses the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm to learn an optimal policy in a given environment.

    The agent consists of an actor network, two critic networks, and their corresponding target networks.
    It also uses a replay buffer to store and sample experiences.

    Attributes:
        gamma (float): The discount factor.
        tau (float): The soft update coefficient for updating the target networks.
        max_actions (np.ndarray): The maximum values for each action.
        min_actions (np.ndarray): The minimum values for each action.
        memory (ReplayBuffer): The replay buffer for storing experiences.
        batch_size (int): The batch size for sampling from the replay buffer.
        learn_step_counter (int): A counter for the number of learning steps.
        time_step (int): A counter for the number of time steps.
        warmup (int): The number of time steps to wait before starting to learn.
        n_actions (int): The number of actions.
        update_actor_interval (int): The interval at which the actor network is updated.
        actor (ActorNetwork): The actor network.
        critic_1 (CriticNetwork): The first critic network.
        critic_2 (CriticNetwork): The second critic network.
        target_actor (ActorNetwork): The target actor network.
        target_critic_1 (CriticNetwork): The first target critic network.
        target_critic_2 (CriticNetwork): The second target critic network.
        noise (float): The standard deviation of the noise added to the actions.

    Args:
        actor_learning_rate (float): The learning rate for the actor network.
        critic_learning_rate (float): The learning rate for the critic networks.
        input_dims (tuple): The dimensions of the input state.
        tau (float): The soft update coefficient for updating the target networks.
        env (gym.Env): The environment in which the agent is trained.
        gamma (float): The discount factor. Defaults to 0.99.
        update_actor_interval (int): The interval at which the actor network is updated. Defaults to 2.
        warmup (int): The number of time steps to wait before starting to learn. Defaults to 1000.
        n_actions (int): The number of actions. Defaults to 2.
        max_size (int): The maximum size of the replay buffer. Defaults to 1000000.
        layer_1_size (int): The size of the first hidden layer in the actor and critic networks. Defaults to 256.
        layer_2_size (int): The size of the second hidden layer in the actor and critic networks. Defaults to 128.
        batch_size (int): The batch size for sampling from the replay buffer. Defaults to 100.
        noise (float): The standard deviation of the noise added to the actions. Defaults to 0.1.
    """
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000, n_actions=2, max_size=1000000, layer_1_size = 256, layer_2_size = 128,batch_size=100, noise=0.1):
        """
        Initializes the Agent object with the given parameters.
        """

        self.gamma = gamma
        self.tau = tau
        self.max_actions = env.action_space.high
        self.min_actions = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval

        # Create the Networks
        # TD3 is a Twin Delayed DDPG. It has single actor, dual critic network, target actor and a target for each critic network.
        # 6 networks in total.

        # Actor Network
        self.actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer_1_size, fc2_dims=layer_2_size, n_actions=n_actions, name="actor", learning_rate=actor_learning_rate)
        # Critic Network
        self.critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer_1_size, fc2_dims=layer_2_size, n_actions=n_actions, name="critic_1", learning_rate=critic_learning_rate)
        self.critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer_1_size, fc2_dims=layer_2_size, n_actions=n_actions, name="critic_2", learning_rate=critic_learning_rate)
        # Target Actor Network
        self.target_actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer_1_size, fc2_dims=layer_2_size, n_actions=n_actions, name="target_actor", learning_rate=actor_learning_rate)
        #Target Critic Network
        self.target_critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer_1_size, fc2_dims=layer_2_size, n_actions=n_actions, name="target_critic_1", learning_rate=critic_learning_rate)
        self.target_critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer_1_size, fc2_dims=layer_2_size, n_actions=n_actions, name="target_critic_2", learning_rate=critic_learning_rate)

        self.noise = noise

        self.update_network_parameters(tau)

    def choose_action(self, observation, validation=False):
        """
        Chooses an action based on the current observation.

        If the current time step is less than the warmup period and it is not validation, it returns a random action sampled from a normal distribution.
        Otherwise, it uses the actor network to predict an action based on the current state.
        Then, it adds noise to the predicted action and clips it to be within the action space.

        Args:
            observation (np.ndarray): The current observation.
            validation (bool): A flag to indicate whether the agent is in validation mode. Defaults to False.

        Returns:
            np.ndarray: The chosen action.
        """
        if self.time_step < self.warmup and validation is False:
            mv = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mv = self.actor.forward(state).to(self.actor.device)

        mv_prime = mv + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        mv_prime = T.clamp(mv_prime, self.min_actions[0], self.max_actions[0])

        self.time_step += 1

        return mv_prime.cpu().detach().numpy()

    def remember(self, state, action, rewards, next_state, done):
        """
        Stores a transition in the replay buffer.

        Args:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            rewards (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        self.memory.store_transition(state, action, rewards, next_state, done)

    def learn(self):
        """
        Updates the actor and critic networks using samples from the replay buffer.

        This method performs the following steps:
        1. Checks if there are enough transitions in the replay buffer to start learning.
           If not, it returns early.
        2. Samples a batch of transitions from the replay buffer.
        3. Converts the samples to PyTorch tensors and moves them to the appropriate device.
        4. Calculates the target Q-values using the target actor and critic networks.
           - The target actions are generated by the target actor network, with added noise.
           - The target Q-values are the minimum of the two target critic networks' Q-values for the next state and target actions.
           - The target Q-values are discounted by the gamma factor and added to the reward.
        5. Calculates the Q-values for the current state and action using the critic networks.
        6. Calculates the critic loss using the mean squared error between the target Q-values and the critic Q-values.
        7. Updates the critic networks by backpropagating the critic loss and applying the optimizer.
        8. If the learn step counter is a multiple of the update actor interval, update the actor network.
           - Calculates the actor loss using the mean of the Q-values from the first critic network for the current state and the actor's predicted action.
           - Updates the actor network by backpropagating the actor loss and applying the optimizer.
           - Updates the target networks using the soft update method.
        """
        if self.memory.mem_counter < self.batch_size * 10:
            return # basically it means that not enough transition to learn

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(next_state)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_actions[0], self.max_actions[0])

        next_q1 = self.target_critic_1.forward(next_state, target_actions)
        next_q2 = self.target_critic_2.forward(next_state, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        next_q1[done] = 0.0
        next_q2[done] = 0.0

        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)

        next_critic_value = T.min(next_q1, next_q2)

        target = reward + self.gamma * next_critic_value
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)

        actor_loss.backward()

        self.actor.optimizer.step()
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """
        Updates the target network parameters using a soft update.

        The target networks are updated towards the main networks by:
        θ_target = τ * θ + (1 - τ) * θ_target
        where θ_target are the target network parameters, θ are the main network parameters, and τ is the soft update parameter.
        """
        if tau == None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + (1 - tau) * target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + (1 - tau) * target_critic_2_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)

    def save_model(self):
        """
        Saves the model weights to the specified file paths.

        This function saves the state dictionaries of the actor, critic, and target networks
        to their respective checkpoint files. This allows the agent to be loaded and resume
        training or evaluation from a previously saved state.
        """
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        print("Saved all the models")


    def load_model(self):
        """
        Loads the model weights from the saved checkpoints.
        It attempts to load the weights for the actor, critics, and target networks.
        If any of the loading operations fail, it prints an error message and continues training from scratch.
        """
        successes = [
            self.actor.load_checkpoint(),
            self.critic_1.load_checkpoint(),
            self.critic_2.load_checkpoint(),
            self.target_actor.load_checkpoint(),
            self.target_critic_1.load_checkpoint(),
            self.target_critic_2.load_checkpoint(),
        ]

        if all(successes):
            print("Successfully loaded all the models")
        else:
            print("Failed to load models. Starting from scratch")
