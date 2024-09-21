import gym
import numpy as np
import torch
from custom_env import CustomEnv
from dqn_agent import DQNAgent
import pandas as pd

# Load and preprocess the dataset
data = pd.read_csv('develop.csv')
target = 'Ins'  # Define your target variable

# Separate features and target
X = data.drop(columns=[target])
y = data[target]

# Preprocess data (you can include your preprocessing steps here)
def preprocess_data(X):
    # Convert categorical variables to dummies, drop first to avoid dummy variable trap
    X = pd.get_dummies(X, drop_first=True)
    return X

# Apply preprocessing
X = preprocess_data(X)

# Initialize the environment with the preprocessed data and target
env = CustomEnv(X, y)  # Pass the data directly

# Initialize the DQN Agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Load the pre-trained model
agent.load("../5 - models/dqn_450.h5")

# Testing the model
state = env.reset()
state = np.reshape(state, [1, state_size])
total_reward = 0

for _ in range(500):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break
    state = next_state

# Print the total reward obtained during the episode
print(f"Total Reward: {total_reward}")