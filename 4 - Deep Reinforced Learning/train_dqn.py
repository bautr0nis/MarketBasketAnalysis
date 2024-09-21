import gym
import numpy as np
import torch
import random
import pandas as pd
from collections import deque
from custom_env import CustomEnv
from dqn_agent import DQNAgent

import os

# Define the correct path for the model directory
model_dir = '../5 - models'  # Adjust this path based on your folder structure

# Check if the directory exists, if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def preprocess_data(X):
    # Convert categorical variables to dummies
    if isinstance(X, pd.DataFrame):
        X = pd.get_dummies(X, drop_first=True)
    # Ensure all data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric data to NaN
    X.fillna(0, inplace=True)  # Replace NaN with 0 or any placeholder
    return X

# Load your dataset
data = pd.read_csv('develop.csv')  # Update path as needed
target = 'Ins'  # Define the target variable

# Separate features and target
X = data.drop(columns=[target])
y = data[target]

# Apply preprocessing to the features
X = preprocess_data(X)

# Initialize environment with preprocessed data
env = CustomEnv(X, y)  # Ensure that CustomEnv accepts preprocessed X and y

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Training loop (rest of the code remains the same)
episodes = 500
batch_size = 32

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size]).astype(np.float32)  # Ensure state is float32
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size]).astype(np.float32)  # Ensure next_state is float32
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 50 == 0:
        agent.save(f"{model_dir}/dqn_{e}.h5")