import numpy as np
from Model.Q_learn import QLearningAgent
from function.visualize import visualize_search

# 从txt文件读取地图
def load_map_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    grid = []
    for line in lines:
        grid.append([int(x) for x in line.strip().split()])
    return grid

# 加载地图并提取障碍物位置
map_file = "scene_data/complex_scene_1.txt"  # 替换为实际地图文件路径
grid_map = load_map_from_file(map_file)
obstacles = [(i, j) for i in range(len(grid_map)) for j in range(len(grid_map[0])) if grid_map[i][j] == 1]
print("Obstacles:", obstacles)

grid_size = (len(grid_map), len(grid_map[0])) #网格大小
start = (0, 0)
goal = (29, 29)

# 状态和动作编码
state_size = grid_size[0] * grid_size[1] #表示环境中状态的数量，用于确定 Q 表的行数。
action_size = 4 #表示智能体可以采取的动作数量，用于确定 Q 表的列数。

agent = QLearningAgent(state_size, action_size)

num_episodes = 1000
for episode in range(num_episodes):
    state = start
    state_index = state[0] * grid_size[1] + state[1]
    done = False
    previous_state = None  # 上一步的状态
    while not done:
        action = agent.choose_action(state_index)
        
        # 根据动作更新状态
        if action == 0:  # 上
            next_state = (state[0] - 1, state[1])
        elif action == 1:  # 下
            next_state = (state[0] + 1, state[1])
        elif action == 2:  # 左
            next_state = (state[0], state[1] - 1)
        elif action == 3:  # 右
            next_state = (state[0], state[1] + 1)

        # 检查边界和障碍物
        if next_state[0] < 0 or next_state[0] >= grid_size[0] or next_state[1] < 0 or next_state[1] >= grid_size[1] or next_state in obstacles:
            next_state = state
            reward = -10
        elif next_state == goal:
            reward = 100
            done = True
        else:
            # 根据目标点远近给予奖励
            distance_before = abs(state[0] - goal[0]) + abs(state[1] - goal[1])
            distance_after = abs(next_state[0] - goal[0]) + abs(next_state[1] - goal[1])
            if distance_after < distance_before:
                reward = 10  # 靠近目标点，给予正奖励
            else:
                reward = -1  # 远离目标点，给予负奖励

        next_state_index = next_state[0] * grid_size[1] + next_state[1]
        
        # 确保智能体不走回头路
        if previous_state is not None and next_state == previous_state:
            # 如果是回头路，则惩罚
            reward = -100
            next_state = state  # 保持当前状态，避免走回头路
            done = True  # 可以结束当前episode（虽然没有到达goal）
        
        agent.update_q_table(state_index, action, reward, next_state_index)
        previous_state = state  # 更新上一步状态
        state = next_state
        state_index = next_state_index

# 测试训练好的模型
path = []
search_steps = []  # 用于记录搜索过程
state = start
state_index = state[0] * grid_size[1] + state[1]
done = False
previous_state = None  # 上一步的状态
while not done:
    action = agent.choose_action(state_index, epsilon=0)
    if action == 0:  # 上
        next_state = (state[0] - 1, state[1])
    elif action == 1:  # 下
        next_state = (state[0] + 1, state[1])
    elif action == 2:  # 左
        next_state = (state[0], state[1] - 1)
    elif action == 3:  # 右
        next_state = (state[0], state[1] + 1)

    # 检查边界和障碍物
    if next_state[0] < 0 or next_state[0] >= grid_size[0] or next_state[1] < 0 or next_state[1] >= grid_size[1] or next_state in obstacles:
        next_state = state
    elif next_state == goal:
        done = True

    # 确保不走回头路
    if previous_state is not None and next_state == previous_state:
        continue  # 如果是回头路，则跳过本次循环，重新选择动作
    
    path.append(next_state)
    # 记录搜索步骤
    search_steps.append({
        'current': next_state,
        'open_set': [],  # 强化学习里没有严格意义的 open_set，这里留空
        'close_set': path.copy(),  # 已经走过的路径作为 close_set
        'g_score': len(path),  # 简单用路径长度作为 g_score
        'f_score': len(path)  # 这里 f_score 暂时和 g_score 一样
    })
    previous_state = state  # 更新上一步状态
    state = next_state
    state_index = next_state[0] * grid_size[1] + next_state[1]

# 创建场景数组
array = np.zeros(grid_size)
for obs in obstacles:
    array[obs] = 1

# 可视化路径及搜索过程
visualize_search(array, path, search_steps, start, goal)
