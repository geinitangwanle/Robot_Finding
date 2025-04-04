from Astar import a_star
from visualize import AStarVisualizer
import matplotlib.pyplot as plt

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
    #第一个为行，第二个为列
    start = (2, 3)
    goal = (4, 7)

    # 调用A*算法
    path, search_steps = a_star(array, start, goal)
    # 检查数据
    print(f"Path: {path}")
    print(f"Search steps: {len(search_steps)}")
    # 可视化搜索过程
    visualizer = AStarVisualizer(array, path, search_steps, start, goal)
    plt.show()