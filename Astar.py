import heapq
from heuristics import heuristic

def a_star(array, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    # 新增：记录搜索过程
    search_steps = []
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        # 记录当前状态
        search_steps.append({
            'current': current,
            'open_set': [item[1] for item in oheap],
            'close_set': close_set.copy(),
            'g_score': gscore.get(current, float('inf')),
            'f_score': fscore.get(current, float('inf'))
        })
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data, search_steps  # 返回路径和搜索过程

        close_set.add(current)
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if 0 <= neighbor[0] < len(array) and 0 <= neighbor[1] < len(array[0]):
                if array[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue

            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
