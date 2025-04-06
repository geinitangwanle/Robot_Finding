from Model.DeepQNN import DQNAgent, MapEnvironment
import torch
# 从txt文件读取地图
def load_map_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    grid = []
    for line in lines:
        grid.append([int(x) for x in line.strip().split()])
    return grid

# 加载地图并提取障碍物位置
map_file = "scene/complex_scene.txt"  # 替换为实际地图文件路径
grid_map = load_map_from_file(map_file)
obstacles = [(i, j) for i in range(len(grid_map)) for j in range(len(grid_map[0])) if grid_map[i][j] == 1]
#print("Obstacles:", obstacles)

grid_size = (len(grid_map), len(grid_map[0])) #网格大小
start = (0, 0)
goal = (9, 9)

# 状态和动作编码
state_size = grid_size[0] * grid_size[1] #表示环境中状态的数量，用于确定 Q 表的行数。
action_size = 4 #表示智能体可以采取的动作数量，用于确定 Q 表的列数。

# 训练函数
def train_dqn(agent, env, num_episodes=1000, max_steps_per_episode=100):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_model(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        
    # 保存模型
    torch.save(agent.model.state_dict(), 'dqn_model.pth')

# 初始化智能体和环境
agent = DQNAgent(state_size, action_size)
env = MapEnvironment(grid_map, start, goal, obstacles, state_size)

# 开始训练
train_dqn(agent, env)

