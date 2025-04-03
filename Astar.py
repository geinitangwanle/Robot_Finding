import heapq
from hn_function import heuristic
'''
A*算法实现
'''
def a_star(array, start, goal):
    # 定义相邻节点的偏移量，分别表示右、左、下、上
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] #允许移动的 四个方向（右、左、下、上）。
    # 已访问节点集合
    close_set = set() # 存储已访问的节点
    # 记录节点的父节点，用于回溯路径
    came_from = {} # 记录路径，用于回溯
    # 从起点到每个节点的实际代价
    gscore = {start: 0} # start 到当前节点的最短路径代价
    # 从起点经过每个节点到目标节点的总代价估计值
    fscore = {start: heuristic(start, goal)}  # 启发式总代价估计
    # 优先队列，用于存储待扩展的节点
    oheap = [] # 优先队列

    # 将起点加入优先队列
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        # 从优先队列中取出 fscore 最小的节点
        current = heapq.heappop(oheap)[1]

        # 如果当前节点是目标节点，回溯路径
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        # 将当前节点标记为已访问
        close_set.add(current)
        for i, j in neighbors:
            # 计算相邻节点的坐标
            neighbor = current[0] + i, current[1] + j
            # 计算从起点经过当前节点到相邻节点的实际代价
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < len(array):
                if 0 <= neighbor[1] < len(array[0]):
                    # 如果相邻节点是障碍物，跳过
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # 越界，跳过
                    continue
            else:
                # 越界，跳过
                continue

            # 如果相邻节点已访问且新的代价更大，跳过
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            # 如果新的代价更小或者相邻节点不在优先队列中，更新信息并加入优先队列
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None