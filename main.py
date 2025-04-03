from Astar import a_star
from visualize import visualize_search

if __name__ == "__main__":
    # 示例场景图矩阵
    array = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    start = (2, 3)
    goal = (4, 7)

    # 调用A*算法
    path, search_steps = a_star(array, start, goal)
    
    # 可视化搜索过程
    visualize_search(array, path, search_steps, start, goal)