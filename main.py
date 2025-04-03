from visualize import visualize_path
from Astar import a_star

if __name__ == "__main__":
    # 示例场景图矩阵，0 表示可通行，1 表示障碍
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

    # 调用 A* 算法搜索路径
    path = a_star(array, start, goal)
    # 可视化路径
    visualize_path(array, path, start, goal)