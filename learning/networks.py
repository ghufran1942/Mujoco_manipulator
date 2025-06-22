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
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128, name="critic", checkpoint_dir="tmp/td3", learning_rate=10e-3, weight_decay=0.005):
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

        print(f"Critic network initialized on device: {self.device}") # print the device the network is initialized on

        self.to(self.device) # move the network to the device

    def forward(self, state, action): # Forward pass for the network
        action_value = self.fc1(T.cat([state, action], dim=1)) # Concatenate state and action tensors along the first dimension
        action_value = F.relu(action_value) # Apply ReLU activation function to the concatenated tensor
        action_value = self.fc2(action_value) # Apply second fully connected layer
        action_value = F.relu(action_value) # Apply ReLU activation function to the output of the second fully connected layer

        q1 = self.q1(action_value) # Apply output layer to the output of the second fully connected layer

        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, learning_rate=10e-3, n_actions=2, name='actor', checkpoint_dir='tmp/td3'):
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

        print(f"Actor network initialized on device: {self.device}") # print the device the network is initialized on

        self.to(self.device) # move the network to the device


    def forward(self, state): # Forward pass for the network
        x = self.fc1(state) # Concatenate state and action tensors along the first dimension
        x = F.relu(x) # Apply ReLU activation function to the concatenated tensor
        x = self.fc2(x) # Apply second fully connected layer
        x = F.relu(x) # Apply ReLU activation function to the output of the second fully connected layer

        x = T.tanh(self.action(x))

        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():

    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000, n_actions=2, max_size=1000000, layer_1_size = 256, layer_2_size = 128,batch_size=100, noise=0.1):

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
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        print("Saved all the models")


    def load_model(self):
        try:
            self.actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
            print("Sucessfully loaded all the models")
        except:
            print("Failed to laod models. Starting from Scratch ")
