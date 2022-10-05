# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:36:35 2020

@author: mehmet
"""



import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
env = gym.make("Taxi-v3")

env.render()

action_size = env.action_space.n
state_size = env.observation_space.n

env.reset()
env.step(env.action_space.sample())[0]

model_only_embedding = Sequential()
model_only_embedding.add(Embedding(500, 6, input_length=1))
model_only_embedding.add(Reshape((6,)))
print(model_only_embedding.summary())

model = Sequential()
model.add(Embedding(500, 10, input_length=1))
model.add(Reshape((10,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(action_size, activation='linear'))
print(model.summary())


memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn_only_embedding = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=500, target_model_update=1e-2, policy=policy)
dqn_only_embedding.compile(Adam(lr=1e-3), metrics=['mae'])
dqn_only_embedding.fit(env, nb_steps=10000, visualize=False, verbose=1, nb_max_episode_steps=99, log_interval=10000)
# dqn_only_embedding.fit(env, nb_steps=100000, visualize=False, verbose=1, nb_max_episode_steps=99, log_interval=100000)

dqn_only_embedding.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=99)

# print('Episode number :', nb_episodes)