import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor #折扣因子，用于权衡当前奖励和未来奖励的重要性。
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, state, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def update_model(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0) # 将状态转换为张量并增加一个维度，batch_size=1
        next_state = torch.FloatTensor(next_state).unsqueeze(0) # 将下一个状态转换为张量并增加一个维度，batch_size=1
        q_values = self.model(state) # 计算当前状态的Q值
        next_q_values = self.model(next_state) # 计算下一个状态的Q值
        max_next_q = torch.max(next_q_values).item() # 获取下一个状态的最大Q值
        target_q = q_values.clone() # 克隆当前状态的Q值
        target_q[0][action] = reward + (1 - done) * self.discount_factor * max_next_q # 更新目标Q值
        self.optimizer.zero_grad() # 清零梯度，防止累加
        loss = self.criterion(q_values, target_q) # 计算损失
        loss.backward() # 反向传播
        self.optimizer.step() # 更新模型参数


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
        reward = 100 if done else -1

        return self._state_to_vector(self.current_state), reward, done

    def _state_to_vector(self, state):
        x, y = state
        vector = np.zeros(self.state_size)
        vector[x * len(self.grid_map[0]) + y] = 1
        return vector