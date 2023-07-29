import torch


class ReplayMemory:
    def __init__(self, max_size:int, batch_size:int, state_size:int, device=torch.device):
        self.max_size = max_size
        self.cur_size = 0
        self.batch_size = batch_size

        self.state_memory = torch.zeros((self.max_size, state_size), device=device)
        self.action_memory = torch.zeros(self.max_size, device=device)
        self.target_q_memory = torch.zeros(self.max_size, device=device)
        self.oldest_pointer = 0

    def log_memory(self, state, action, target_q):
        if self.cur_size < self.max_size:
            self.state_memory[self.cur_size, :] = state
            self.action_memory[self.cur_size] = action
            self.target_q_memory[self.cur_size] = target_q
            self.cur_size += 1
        else:
            self.state_memory[self.oldest_pointer, :] = state
            self.action_memory[self.oldest_pointer] = action
            self.target_q_memory[self.oldest_pointer] = target_q
            self.oldest_pointer += 1
        
        if self.oldest_pointer >= self.max_size:
            self.oldest_pointer = 0

    def get_batch(self, size=None):
        if size is None:
            size = self.batch_size
        idxs = torch.randperm(self.cur_size)
        idxs = idxs[:size]

        return self.state_memory[idxs, :], self.action_memory[idxs], self.target_q_memory[idxs]


class DQNAgent:
    def __init__(self, 
                    state_shape, 
                    n_actions, 
                    replay_memory_size,
                    max_episodes,
                    neural_net_class,
                    neural_net_args,
                    environment) -> None:

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.replay_memory_size = replay_memory_size
        self.max_episodes = max_episodes
        self.env = environment

        # Making decision and target nets
        self.decision_net = neural_net_class(neural_net_args)
        self.target_net = neural_net_class(neural_net_args)
        self.target_net.load_state_dict(self.decision_net.state_dict())

    def train(self):
        for episode in self.max_episodes:
            self.env.reset()
            is_done = False

            while not(is_done):
                cur_state = self.env.get_cur_state()
                

            


