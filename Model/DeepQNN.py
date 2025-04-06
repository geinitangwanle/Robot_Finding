import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 增加隐藏层神经元数量
        self.fc1 = nn.Linear(state_size, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)  # 新增一层隐藏层
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # 新增隐藏层的前向传播
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.9,prev_state=None):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.prev_state = prev_state  # 用于记录上一个状态

    def choose_action(self, state):
        if self.prev_state is not None:
            # 计算回到上一个状态的动作索引
            x, y = state
            px, py = self.prev_state
            if x == px + 1:  # 上一个状态在下方，禁止选择向上的动作（索引为 0）
                prohibited_action = 0
            elif x == px - 1:  # 上一个状态在上方，禁止选择向下的动作（索引为 1）
                prohibited_action = 1
            elif y == py + 1:  # 上一个状态在左方，禁止选择向左的动作（索引为 2）
                prohibited_action = 2
            elif y == py - 1:  # 上一个状态在右方，禁止选择向右的动作（索引为 3）
                prohibited_action = 3
            else:
                prohibited_action = None

            if prohibited_action is not None:
                q_values = self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
                q_values[prohibited_action] = float('-inf')  # 将禁止的动作的 Q 值设为负无穷
                action = np.argmax(q_values)
                return action

        if np.random.uniform(0, 1) < 0.1:
            return np.random.choice(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def update_model(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.model(state)
        next_q_values = self.model(next_state)
        max_next_q = torch.max(next_q_values).item()
        target_q = q_values.clone()
        target_q[0][action] = reward + (1 - done) * self.discount_factor * max_next_q
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q)
        loss.backward()
        self.optimizer.step()


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
        elif (new_x, new_y) in self.obstacles:
            reward = -10  # 撞到障碍物给予较大负奖励
        else:
            # 根据距离目标点的远近给予奖励
            reward = -1 + (np.linalg.norm(np.array((x, y)) - np.array(self.goal)) - current_distance) * 10  

        return self._state_to_vector(self.current_state), reward, done

    def _state_to_vector(self, state):
        x, y = state
        vector = np.zeros(self.state_size)
        vector[x * len(self.grid_map[0]) + y] = 1
        return vector