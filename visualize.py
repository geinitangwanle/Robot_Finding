import matplotlib.pyplot as plt


def visualize_path(array, path, start, goal):
    # 显示场景图
    plt.imshow(array, cmap='gray')
    # 标记起点
    plt.plot(start[1], start[0], 'go')
    # 标记终点
    plt.plot(goal[1], goal[0], 'ro')

    if path:
        # 反转路径，因为回溯得到的路径是从终点到起点的
        path = path[::-1]
        x = [p[1] for p in path]
        y = [p[0] for p in path]
        # 绘制路径
        plt.plot(x, y, 'b-')

    # 显示图形
    plt.show()