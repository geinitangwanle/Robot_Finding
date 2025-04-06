from Model.DeepQNN import DQNAgent, MapEnvironment
from Model.Astar import a_star
import numpy as np
import torch
import random
import os

# 从txt文件读取地图
def load_map_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    grid = []
    for line in lines:
        grid.append([int(x) for x in line.strip().split()])
    return grid

# 从scene_data文件夹中加载所有地图文件
scene_data_folder = "scene_data"
map_files = [os.path.join(scene_data_folder, file) for file in os.listdir(scene_data_folder) if file.endswith(".txt")]

# 训练函数
def train_dqn(agent, num_episodes=500, max_steps_per_episode=100):
    global_step = 0
    for episode in range(num_episodes):
        map_file = "scene_data/complex_scene_1.txt"
        grid_map = load_map_from_file(map_file)
        obstacles = [(i, j) for i in range(len(grid_map)) for j in range(len(grid_map[0])) if grid_map[i][j] == 1]
        grid_size = (len(grid_map), len(grid_map[0]))
        start = (0, 0)
        goal = (grid_size[0] - 1, grid_size[1] - 1)
        state_size = grid_size[0] * grid_size[1]
        env = MapEnvironment(grid_map, start, goal, obstacles, state_size)

        # 使用A*算法获取最优路径
        path, _ = a_star(grid_map, start, goal)
        optimal_path = set(path)

        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state, epsilon=0.1)  # 传递上一个状态
            next_state, reward, done = env.step(action)

            # 获取当前状态对应的坐标
            current_state_index = np.argmax(state)
            current_x = current_state_index // len(grid_map[0])
            current_y = current_state_index % len(grid_map[0])
            current_state_coord = (current_x, current_y)

            # 根据是否在最优路径上调整奖励
            if current_state_coord in optimal_path:
                reward += 1  # 给予额外奖励

            agent.remember(state, action, reward, next_state, done)
            agent.update_model()
            agent.update_target_model(global_step)
            state = next_state
            total_reward += reward
            global_step += 1

            if done:
                break

        
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # 保存模型
    torch.save(agent.model.state_dict(), 'dqn_model_1.pth')

# 初始化智能体
state_size = len(load_map_from_file(map_files[0])) * len(load_map_from_file(map_files[0])[0])
action_size = 4
agent = DQNAgent(state_size, action_size)

# 开始训练
train_dqn(agent)