import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.9,
                 batch_size=64, memory_size=10000, target_update_freq=100, weight_decay=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.learning_rate_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < 0.2:
            return np.random.choice(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in minibatch])

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.discount_factor * next_q_values

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q)
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()

    def update_target_model(self, step):
        if step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

# 定义环境类
class MapEnvironment:
    def __init__(self, grid_map, start, goal, obstacles, state_size):
        self.grid_map = grid_map
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.current_state = start
        self.state_size = state_size

    def reset(self):
        self.current_state = self.start
        return self._state_to_vector(self.current_state)

    def step(self, action):
        x, y = self.current_state
        if action == 0:  # 上
            new_x = max(0, x - 1)
            new_y = y
        elif action == 1:  # 下
            new_x = min(len(self.grid_map) - 1, x + 1)
            new_y = y
        elif action == 2:  # 左
            new_x = x
            new_y = max(0, y - 1)
        elif action == 3:  # 右
            new_x = x
            new_y = min(len(self.grid_map[0]) - 1, y + 1)

        if (new_x, new_y) in self.obstacles:
            new_x, new_y = x, y

        self.current_state = (new_x, new_y)
        done = self.current_state == self.goal

        # 计算到目标点的距离
        current_distance = np.linalg.norm(np.array(self.current_state) - np.array(self.goal))

        if done:
            reward = 100
        else:
            # 根据距离目标点的远近给予奖励
            reward = -1 + (np.linalg.norm(np.array((x, y)) - np.array(self.goal)) - current_distance)

        return self._state_to_vector(self.current_state), reward, done

    def _state_to_vector(self, state):
        x, y = state
        vector = np.zeros(self.state_size)
        vector[x * len(self.grid_map[0]) + y] = 1
        return vector