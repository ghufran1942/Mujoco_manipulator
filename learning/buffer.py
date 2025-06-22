import numpy as np

# (State, Action, Reward, Next State, Done=bool)

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size # Maximum size of the buffer, You don't want the buffer to grow indefinitely, so we need to limit its size
        self.mem_counter = 0 # Initialize the counter to keep track of the number of elements in the buffer
        self.state_memory = np.zeros((self.mem_size, *input_shape)) # Initialize the state memory to store the states of the environment\
        self.next_state_memory = np.zeros((self.mem_size, *input_shape)) # Initialize the new state memory to store the next states of the environment
        print(f"Initialized ReplayBuffer with max_size={self.mem_size}, input_shape={input_shape}, n_actions={n_actions}")
        # print(f"Inside Buffer: n_actions type={type(n_actions)}")
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size
        print(f"Inside Buffer: index={index}")

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        state = self.state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        next_state = self.next_state_memory[batch]
        done = self.terminal_memory[batch]

        return state, action, reward, next_state, done
