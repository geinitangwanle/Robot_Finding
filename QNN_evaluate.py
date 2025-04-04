from Model.DeepQNN import DQNAgent, MapEnvironment
from function.visualize import visualize_search
import torch
import numpy as np
# 从txt文件读取地图
def load_map_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    grid = []
    for line in lines:
        grid.append([int(x) for x in line.strip().split()])
    return grid

# 加载地图并提取障碍物位置
map_file = "scene/custom_scene.txt"  # 替换为实际地图文件路径
grid_map = load_map_from_file(map_file)
obstacles = [(i, j) for i in range(len(grid_map)) for j in range(len(grid_map[0])) if grid_map[i][j] == 1]
#print("Obstacles:", obstacles)

grid_size = (len(grid_map), len(grid_map[0])) #网格大小
start = (2, 3)
goal = (4, 7)

# 状态和动作编码
state_size = grid_size[0] * grid_size[1] #表示环境中状态的数量，用于确定 Q 表的行数。
action_size = 4 #表示智能体可以采取的动作数量，用于确定 Q 表的列数。


# 初始化智能体
agent = DQNAgent(state_size, action_size)
# 加载模型参数
agent.model.load_state_dict(torch.load('dqn_model.pth'))
agent.model.eval()  # 设置为评估模式


# 初始化环境
env = MapEnvironment(grid_map, start, goal, obstacles, state_size)

# 重置环境
state = env.reset()
path = [env.current_state]
search_steps = []
done = False
max_steps = 100
step_count = 0

while not done and step_count < max_steps:
    action = agent.choose_action(state, epsilon=0)  # 探索率设为 0，只进行利用
    next_state, _, done = env.step(action)
    path.append(env.current_state)
    # 记录搜索步骤
    search_steps.append({
        'current': env.current_state,
        'open_set': [],  # DQN 没有严格意义的 open_set，这里留空
        'close_set': path.copy(),  # 已经走过的路径作为 close_set
        'g_score': len(path),  # 简单用路径长度作为 g_score
        'f_score': len(path)  # 这里 f_score 暂时和 g_score 一样
    })
    state = next_state
    step_count += 1

# 可视化搜索过程
# 创建场景数组
array = np.array(grid_map)
# 可视化路径
visualize_search(array, path, search_steps, start, goal)