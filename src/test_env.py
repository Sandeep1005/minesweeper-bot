import gym
import random


env = gym.make('mibexx_gym_minesweeper:mibexx-gym-minesweeper-v0')
env.reset()

env.render()

env.step(random.choice(env.action_space))

