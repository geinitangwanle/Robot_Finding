import matplotlib.pyplot as plt


def visualize_path(array, path, start, goal):
    # 将array中的值映射为合适的颜色数据，这里假设0是白色（可通行），1是蓝色（障碍）
    cmap = plt.cm.get_cmap('Blues')
    norm = plt.Normalize(vmin=0, vmax=1)
    colored_array = cmap(norm(array))

    # 显示场景图
    plt.imshow(colored_array, origin='upper')

    # 标记起点为红色
    plt.plot(start[1], start[0], 'rs', markersize=27)
    # 标记终点为灰色
    plt.plot(goal[1], goal[0], 'gs', markersize=27, color='gray')

    if path:
        # 反转路径，因为回溯得到的路径是从终点到起点的
        path = path[::-1]
        x = [p[1] for p in path]
        y = [p[0] for p in path]
        # 绘制路径
        plt.plot(x, y, 'b-')

    # 显示图形
    plt.show()