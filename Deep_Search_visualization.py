from function.visualize import visualize_search
import numpy as np
from Model.Q_learn import QLearningAgent


# 获取智能体的路径
path = []
state = start
state_index = state[0] * grid_size[1] + state[1]
done = False
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

    if next_state[0] < 0 or next_state[0] >= grid_size[0] or next_state[1] < 0 or next_state[1] >= grid_size[1] or next_state in obstacles:
        next_state = state
    elif next_state == goal:
        done = True

    path.append(next_state)
    state = next_state
    state_index = next_state[0] * grid_size[1] + next_state[1]

# 创建场景数组
array = np.zeros(grid_size)
for obs in obstacles:
    array[obs] = 1

# 可视化路径
visualize_search(array, path, [], start, goal)