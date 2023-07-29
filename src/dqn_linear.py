import torch.nn as nn
import torch
import numpy as np
import random
import mlflow
import mlflow.pytorch
import os
from dataclasses import dataclass, asdict

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

from mine_sweeper_env import MinesweeperEnv, is_win


class ReplayMemory:
    def __init__(self, max_size, batch_size) -> None:
        self.max_size = max_size
        self.batch_size = batch_size

        self.memory = list()
        self.pointer = 0
        
    def log_memory(self, memory):
        if len(self.memory) < self.max_size:
            self.memory.append(memory)
        else:
            self.memory[self.pointer] = memory
            self.pointer += 1
            if self.pointer >= len(self.memory):
                self.pointer = 0
    
    def get_sample(self, size:int=0):
        if size == 0:
            size = self.batch_size
        return random.sample(self.memory, size)



class Agent(nn.Module):
    def __init__(self, board_x, board_y) -> None:
        super().__init__()
        self.board_x = board_x
        self.board_y = board_y

        input_size = board_x*board_y
        hidden_size = 100
        output_size = board_x*board_y

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = input.reshape(input.shape[0], self.board_x*self.board_y)
        return self.linear_2(self.relu(self.linear_1(input))).reshape((input.shape[0], self.board_x, self.board_y))


@dataclass
class Parameters:
    BATCH_SIZE: int = 512
    MAX_REPLAY_MEMEORY_SIZE: int = 10000
    BOARD_WIDTH: int = 5 # width = height
    BOARD_HEIGHT: int = 5
    N_MINES: int = 5

    MAX_EPISODES: int = 10000
    LEARNING_RATE: float = 1E-3
    MOMENTUM: float = 0.9

    EPSILON_MAX: float = 0.9
    EPSILON_MIN: float = 0.01
    EPSILON_DECAY: float = 0.9

    MIN_REPLAY_MEMORY_SIZE_TO_START_TRAINING: int = 1000
    N_BATCHES_TO_TRAIN_PER_STEP: int = 10


def train(params:Parameters):
    exp = mlflow.set_experiment("DQN_linear")
    mlflow.start_run(experiment_id=exp.experiment_id)

    mlflow.log_params(asdict(params))

    epsilon = params.EPSILON_MAX

    env = MinesweeperEnv(board_size=params.BOARD_HEIGHT, num_mines=params.N_MINES)
    rep_mem = ReplayMemory(max_size=params.MAX_REPLAY_MEMEORY_SIZE, batch_size=params.BATCH_SIZE)

    decision_net = Agent(board_x=params.BOARD_HEIGHT, board_y=params.BOARD_WIDTH)
    # target_net = Agent(board_x=params.BOARD_HEIGHT, board_y=params.BOARD_WIDTH)
    # target_net.load_state_dict(decision_net.state_dict())
    # target_net.eval()

    optimizer = torch.optim.SGD(params=decision_net.parameters(), lr=params.LEARNING_RATE, momentum=params.MOMENTUM)
    loss = torch.nn.functional.mse_loss

    total_steps = 0
    for episode in range(params.MAX_EPISODES):
        is_episode_done = False
        steps_in_episode = 0
        while not(is_episode_done):
            observation = env.my_board
            observation = torch.tensor(observation).float()[None, None, :]

            if random.uniform(0, 1) < epsilon:
                action = [random.randint(0, params.BOARD_HEIGHT-1), random.randint(0, params.BOARD_WIDTH-1)]
            else:
                with torch.no_grad():
                    action = decision_net(observation)
                    action = torch.argmax(action).item()
                    action_x = action // params.BOARD_WIDTH
                    action_y = action % params.BOARD_WIDTH
                    action = [action_x, action_y]

            _, reward, is_episode_done, _ = env.step(action=action)
            rep_mem.log_memory((observation, action, reward))
            steps_in_episode += 1

            # Training
            if len(rep_mem.memory) > params.MIN_REPLAY_MEMORY_SIZE_TO_START_TRAINING:
                decision_net.train()
                for _ in range(params.N_BATCHES_TO_TRAIN_PER_STEP):
                    optimizer.zero_grad()

                    batch = rep_mem.get_sample()
                    observations_list = [item[0] for item in batch]
                    actions_list = [item[1] for item in batch]
                    rewards_list = [item[2] for item in batch]

                    observations_list = torch.concat(observations_list, dim=0)
                    actions_x_list = [item[0] for item in actions_list]
                    actions_y_list = [item[1] for item in actions_list]
                    rewards_list = torch.tensor(rewards_list).float()

                    predicted_rewards = torch.squeeze(decision_net(observations_list), dim=1)
                    loss_value = loss(predicted_rewards[list(range(len(rewards_list))), actions_x_list, actions_y_list].flatten(), rewards_list)
                    loss_value.backward()
                    optimizer.step()

                decision_net.eval()

            total_steps += 1

        # Updating epsilon greedy policy
        epsilon = (epsilon - params.EPSILON_MIN)*params.EPSILON_DECAY + params.EPSILON_MIN

        mlflow.pytorch.log_model(decision_net, artifact_path='Agent_after_episode_'+str(episode))
        mlflow.log_metric('Win_status', is_win(env), step=episode)
        mlflow.log_metric('Steps_in_episode', total_steps, step=episode)

        print("Episode:", episode, "Win_status:", is_win(env))

        env.reset()


if __name__ == '__main__':
    train(Parameters())