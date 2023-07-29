import torch.nn as nn
import torch
import numpy as np
import random
import mlflow
import mlflow.pytorch
import os

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


class Actor(nn.Module):
    def __init__(self, board_width, board_height) -> None:
        super().__init__()
        input_size = board_width * board_height
        output_size = 2
        hidden_size = 200
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        output = self.sigmoid1(self.linear1(input))
        output = self.sigmoid2(self.linear2(output) / 10)
        return output


class Critic(nn.Module):
    def __init__(self, board_width, board_height) -> None:
        super().__init__()
        input_size = board_width * board_height + 2
        output_size = 1
        hidden_size = 200

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        output = torch.concat([states, actions], dim=1)
        output = self.relu(self.linear1(output))
        output = self.linear2(output)
        return output
        
    


def train():
    exp = mlflow.set_experiment("Actor_critic")
    mlflow.start_run(experiment_id=exp.experiment_id)

    # Start of Hyperparameters
    BOARD_SIZE = 10
    NUMBER_OF_MINES = 2

    LEARNING_RATE = 1E-5
    BETAS = (0.9, 0.999)
    MAX_ITERATIONS_FOR_CONVERGENCE = 50
    CONVERGENCE_LIMIT = -1 # Negative value for ignoring early stopping

    MAX_REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 512
    MIN_REPLAY_MEMORY_SIZE_TO_START_TRAINING = 1000
    UPDATE_TARGET_NET_PER_STEPS = 100

    SD_MAX_FOR_EXPLORATION = 0.3
    SD_MIN_FOR_EXPLORATION = 0.0001
    MEAN_FOR_EXPLORATION = 0
    SD_DECAY_FOR_EXPLORATION = 0.9

    # Exploration using epsilon decay
    EPSILON_MAX = 0.5
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.9

    MAX_EPISODES = 10000

    TD_LAMBDA = 10
    DISCOUNT_FACTOR = 0.99
    # End of hyperparameters

    sd_explore = SD_MAX_FOR_EXPLORATION
    epsilon = EPSILON_MAX

    # Logging hyperparameters using mlflow
    mlflow.log_param('BOARD_SIZE', BOARD_SIZE)
    mlflow.log_param('NUMBER_OF_MINTES', NUMBER_OF_MINES)
    mlflow.log_param('LEARNING_RATE', LEARNING_RATE)
    mlflow.log_param('BETAS', BETAS)
    mlflow.log_param('MAX_ITERATIONS_FOR_CONVERGENCE', MAX_ITERATIONS_FOR_CONVERGENCE)
    mlflow.log_param('MAX_REPLAY_MEMORY_SIZE', MAX_REPLAY_MEMORY_SIZE)
    mlflow.log_param('BATCH_SIZE', BATCH_SIZE)
    mlflow.log_param('MIN_REPLAY_MEMORY_SIZE_TO_START_TRAINING', MIN_REPLAY_MEMORY_SIZE_TO_START_TRAINING)
    mlflow.log_param('UPDATE_TARGET_NET_PER_STEPS', UPDATE_TARGET_NET_PER_STEPS)
    mlflow.log_param('MAX_EPISODES', MAX_EPISODES)
    mlflow.log_param('TD_LAMBDA', TD_LAMBDA)
    mlflow.log_param('DISCOUNT_FACTOR', DISCOUNT_FACTOR)
    # End of logging hyperparameters
    
    # Initializing networks
    actor_net = Actor(board_height=BOARD_SIZE, board_width=BOARD_SIZE)
    critic_net = Critic(board_height=BOARD_SIZE, board_width=BOARD_SIZE)
    target_actor_net = Actor(board_height=BOARD_SIZE, board_width=BOARD_SIZE)
    target_critic_net = Critic(board_height=BOARD_SIZE, board_width=BOARD_SIZE)
    target_actor_net.load_state_dict(actor_net.state_dict())
    target_critic_net.load_state_dict(critic_net.state_dict())
    target_actor_net.eval()
    target_critic_net.eval()

    # Loss function and optimizer
    loss_function = torch.nn.functional.mse_loss 
    optimizer_actor = torch.optim.Adam(params=actor_net.parameters(), lr=LEARNING_RATE, betas=BETAS)
    optimizer_critic = torch.optim.Adam(params=critic_net.parameters(), lr=LEARNING_RATE, betas=BETAS)

    # Preparing environment, and replay memory
    env = MinesweeperEnv(board_size=BOARD_SIZE, num_mines=NUMBER_OF_MINES)
    replay_memory = ReplayMemory(max_size=MAX_REPLAY_MEMORY_SIZE, batch_size=BATCH_SIZE)

    total_step_count = 0
    for episode in range(MAX_EPISODES):
        is_episode_done = False
        total_reward = 0

        steps_in_this_epoch = 0
        while is_episode_done is False:
            # Go through the environment for labmda number of steps
            states_list = list()
            actions_list = list()
            rewards_list = list()

            td_lambda_count = 0
            while td_lambda_count < TD_LAMBDA and not(is_episode_done):
                # Get an observation
                observation = torch.tensor(env.my_board).flatten().float()
                states_list.append(observation)
               
                # Prediction of action using actor net
                actor_net.eval()
                action = actor_net(observation[None, :]).flatten().detach()
                
                # # Add noise to action here for exploration purpose
                # noise = torch.normal(mean=torch.ones_like(action)*MEAN_FOR_EXPLORATION, std=sd_explore)
                # final_action = action + noise
                # # Reamaking noise if it goes out of bounds
                # while (torch.sum(final_action < 0).item() + torch.sum(final_action > 1).item()) > 0:
                #     noise = torch.normal(mean=torch.ones_like(final_action)*MEAN_FOR_EXPLORATION, std=sd_explore)
                #     final_action = action + noise
                # action = final_action

                # Using epsilon greedy policy
                if random.uniform(0, 1) < epsilon:
                    action = torch.rand_like(action)

                # Appending the action
                actions_list.append(action.detach())

                # Performing the action in the environment
                final_action = action * BOARD_SIZE
                new_state, reward, is_episode_done, _ = env.step(final_action.detach().numpy())
                
                total_reward += reward
                rewards_list.append(reward)

                td_lambda_count += 1
            
            # Saving the observations obtained
            target_q_values = [0]*len(states_list)
            if is_episode_done:
                last_target = 0
            else:
                with torch.no_grad():
                    output_of_actor = target_actor_net(torch.tensor(new_state).float().flatten()[None, :])
                    last_target = target_critic_net(torch.tensor(new_state).float().flatten()[None, :], output_of_actor).item()
            target_q_values [-1] = rewards_list[-1] + DISCOUNT_FACTOR * last_target

            # Calculating all target q values
            for i in range(len(target_q_values)-2, -1, -1):
                target_q_values[i] = rewards_list[i] + DISCOUNT_FACTOR * target_q_values[i+1]

            for item in zip(states_list, actions_list, target_q_values):
                replay_memory.log_memory(item)

            # Training the nets
            if len(replay_memory.memory) > MIN_REPLAY_MEMORY_SIZE_TO_START_TRAINING:
                # Training critic net first
                critic_net.train()
                prev_loss = 0
                for i in range(MAX_ITERATIONS_FOR_CONVERGENCE):
                    optimizer_critic.zero_grad()

                    batch = replay_memory.get_sample()
                    
                    states_list = [item[0] for item in batch]
                    actions_list = [item[1] for item in batch]
                    target_q_values = [item[2] for item in batch]
                    
                    states_list = torch.stack(states_list)
                    actions_list = torch.stack(actions_list)
                    target_q_values = torch.tensor(target_q_values).reshape(len(target_q_values), 1)
                    
                    predicted_q_values = critic_net(states_list, actions_list)

                    loss = loss_function(predicted_q_values, target_q_values)
                    loss.backward()
                    optimizer_critic.step()

                    # Check for convergence for an early exit
                    change_in_loss = abs(prev_loss - loss.item()) / loss.item()
                    if change_in_loss < CONVERGENCE_LIMIT:
                        print('Exiting due to early convergence!!!', end='\r')
                        break
                    else:
                        prev_loss = loss.item()
                        print('Steps in training of critic', i, end='\r')
                
                critic_net.eval()

                # Now, training actor net to maximize Q value.
                prev_loss = 0
                actor_net.train()
                for i in range(MAX_ITERATIONS_FOR_CONVERGENCE):
                    optimizer_actor.zero_grad()
                    batch = replay_memory.get_sample()
                    
                    states_list = [item[0] for item in batch]
                    states_list = torch.stack(states_list)

                    output_of_actor = actor_net(states_list)
                    final_q_values = critic_net(states_list, output_of_actor)
                    loss = -1 * torch.sum(final_q_values)
                    loss.backward()
                    optimizer_actor.step()

                    # Check for convergence for an early exit
                    change_in_loss = abs(prev_loss - loss.item()) / loss.item()
                    if change_in_loss < CONVERGENCE_LIMIT:
                        print('Exiting due to early convergence!!!', end='\r')
                        break
                    else:
                        prev_loss = loss.item()
                        print('Steps in training of actor', i, end='\r')
                
                actor_net.eval()
            
            steps_in_this_epoch += 1
            total_step_count += 1
            if total_step_count % UPDATE_TARGET_NET_PER_STEPS == 0:
                target_actor_net.load_state_dict(actor_net.state_dict())
                target_critic_net.load_state_dict(critic_net.state_dict())
            
        

        print(f"Episode: {episode:5d}" ,f"Total reward: {total_reward:.2f}")

        # Saving metrics and models after every epoch
        mlflow.log_metric('total_reward', total_reward, step=episode)
        if total_reward > -50:
            # Evaluating win rate
            print('Evaluating....', end='\r')
            win_rate = evaluate(actor=actor_net, env=env)
            mlflow.log_metric('Win rate', win_rate, step=episode)
            mlflow.pytorch.log_model(actor_net, artifact_path='Actor_net_after_episode_'+str(episode))
            mlflow.pytorch.log_model(critic_net, artifact_path='Critic_net_after_episode_'+str(episode))

        # Resetting the environment
        env.reset()
        is_episode_done = False

        # Exploration decay
        #sd_explore = (sd_explore - SD_MIN_FOR_EXPLORATION) * SD_DECAY_FOR_EXPLORATION + SD_MIN_FOR_EXPLORATION
        epsilon = (epsilon - EPSILON_MIN)*EPSILON_DECAY + EPSILON_MIN

    mlflow.end_run()


def evaluate(actor:Actor, env:MinesweeperEnv) -> float:
    """Evaluates and returns the win rate of the model"""
    actor.eval()
    env.reset()
    total_tries = 0
    total_wins = 0
    for i in range(1000):
        print('Try in evaluate', i, end='\r')
        is_episode_done = False
        while not(is_episode_done):
            obs = env.my_board
            with torch.no_grad():
                action = actor(torch.tensor(obs).float().flatten()[None, :]).flatten()
            _, _, is_episode_done, _ = env.step(action=action.flatten().detach().numpy())
        if is_win(env.my_board):
            total_wins += 1
        total_tries += 1

        env.reset()
    
    win_rate = total_wins / total_tries
    return win_rate
    
if __name__ == '__main__':
    train()
