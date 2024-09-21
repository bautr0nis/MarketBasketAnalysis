import gym
import pandas as pd
import numpy as np
from gym import spaces


class CustomEnv(gym.Env):
    def __init__(self, csv_path=None, target_column=None, X=None, y=None):
        super(CustomEnv, self).__init__()

        # If csv_path is provided, read the CSV file
        if csv_path is not None and target_column is not None:
            self.data = pd.read_csv(csv_path)
            self.target = self.data[target_column]
            self.data = self.data.drop(columns=[target_column])
        # If X and y are provided, use them directly
        elif X is not None and y is not None:
            self.data = X
            self.target = y
        else:
            raise ValueError("Please provide either csv_path and target_column, or X and y.")

        self.current_index = 0

        # Define action and observation space
        n_features = self.data.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.current_index = 0
        return self.data.iloc[self.current_index].values

    def step(self, action):
        reward = self.target.iloc[self.current_index]
        self.current_index += 1
        done = self.current_index >= len(self.data)
        next_state = self.data.iloc[self.current_index].values if not done else np.zeros(self.data.shape[1])
        return next_state, reward, done, {}