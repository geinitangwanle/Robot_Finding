import numpy as np
from Model.Astar import a_star
from function.visualize import AStarVisualizer
import matplotlib.pyplot as plt


def read_scene_from_file(file_path):
    """
    从文本文件中读取场景图
    """
    scene = []
    with open(file_path, 'r') as file:
        for line in file:
            # 移除多余空格并过滤非数字字符
            row = [int(cell) for cell in line.strip() if cell.isdigit()]
            scene.append(row)
    return np.array(scene)

if __name__ == "__main__":
    file_path = "scene/complex_scene.txt"  # 假设你的地图文件名为 complex_scene.txt
    array = read_scene_from_file(file_path)

    # 设置起点和终点
    start = (0, 0)  # 起点坐标
    goal =  (29,29)  # 终点坐标

    # 调用A*算法
    path, search_steps = a_star(array, start, goal)

    # 检查数据
    print(f"Path: {path}")
    print(f"Search steps: {len(search_steps)}")

    # 可视化搜索过程
    visualizer = AStarVisualizer(array, path, search_steps, start, goal)
    plt.show()