import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation

class AStarVisualizer:
    def __init__(self, array, path, search_steps, start, goal):
        self.array = array
        self.path = path
        self.search_steps = search_steps
        self.start = start
        self.goal = goal
        self.current_frame = 0
        

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(0, len(array[0]))
        self.ax.set_ylim(0, len(array))
        self.ax.set_aspect('equal')

        # 准备所有帧
        self.prepare_frames()

        # 初始化图形元素
        self.init_elements()

        # 显示初始帧
        self.update_frame(0)
        self.ax.set_title(" Algorithm Visualization (Auto-playing)")

        # 创建动画
        self.ani = FuncAnimation(self.fig, self.update_frame, frames=len(self.frames), interval=100, repeat=True)

    def prepare_frames(self):
        """准备所有动画帧"""
        self.frames = []

        # 添加搜索过程帧
        for step in self.search_steps:
            frame = np.copy(self.array)
            # 开放列表 
            for node in step['open_set']:
                frame[node[0], node[1]] = 4

            # 关闭列表 
            for node in step['close_set']:
                frame[node[0], node[1]] = 5

            # 当前节点
            frame[step['current'][0], step['current'][1]] = 2

            # 起点和终点
            frame[self.start[0], self.start[1]] = 3
            frame[self.goal[0], self.goal[1]] = 7

            self.frames.append(frame)

        # 添加路径帧
        if self.path:
            path_frame = np.copy(self.array)

            # 绘制路径 
            for node in self.path:
                path_frame[node[0], node[1]] = 6

            # 起点和终点
            path_frame[self.start[0], self.start[1]] = 3
            path_frame[self.goal[0], self.goal[1]] = 7

            # 添加多帧使路径显示更久
            self.frames.extend([path_frame] * 5)

    def init_elements(self):
        self.elements = []
        for i in range(len(self.array)):
            row = []
            for j in range(len(self.array[0])):
                rect = Rectangle((j, len(self.array) - i - 1), 1, 1, edgecolor='gray', facecolor='none')
                self.ax.add_patch(rect)
                circle = Circle((j + 0.5, len(self.array) - i - 0.5), 0.2, color='none')
                self.ax.add_patch(circle)
                text = self.ax.text(j + 0.5, len(self.array) - i - 0.5, '', color='black', ha='center', va='center', fontsize=10, weight='bold')
                row.extend([rect, circle, text])
            self.elements.append(row)

    def update_frame(self, frame_num):
        self.current_frame = frame_num
        frame = self.frames[frame_num]

        for i in range(len(frame)):
            for j in range(len(frame[0])):
                value = frame[i][j]
                rect, circle, text = self.elements[i][j * 3:(j + 1) * 3]

                rect.set_facecolor('none')
                circle.set_color('none')
                text.set_text('')

                if value == 1:  # 障碍物，绘制黑色矩形
                    rect.set_facecolor('lightblue')
                elif value == 2:  # 当前节点,绘制蓝色圆形，字母R
                    circle.set_color('blue')
                    text.set_text('R')
                    text.set_color('white')
                elif value == 3:  # 起点
                    rect.set_facecolor('red')
                    text.set_text('S')
                    text.set_color('black')
                    if frame_num == 0:
                        circle.set_color('blue')
                        text.set_text('R')
                        text.set_color('white')
                elif value == 4:  # 开放列表，绘制黄色圆形
                    circle.set_color('yellow')
                elif value == 5:  # 关闭列表，绘制紫色圆形
                    circle.set_color('purple')
                elif value == 6:  # 路径，绘制绿色矩形，字母P
                    circle.set_color('green')
                    text.set_text('P')
                    text.set_color('black')
                elif value == 7:  # 终点，绘制灰色矩形，字母E
                    rect.set_facecolor('gray')
                    text.set_text('E')
                    text.set_color('black')

        # 更新标题显示当前帧信息
        if frame_num < len(self.search_steps):
            step = self.search_steps[frame_num]
            info = f"Step {frame_num + 1}/{len(self.search_steps)} - Current: {step['current']} - Open: {len(step['open_set'])} - Closed: {len(step['close_set'])}"
        else:
            info = "Path found! Animation will restart."

        self.ax.set_title(f" Algorithm Visualization\n{info}")

        self.fig.canvas.draw()


def visualize_search(array, path, search_steps, start, goal):
    visualizer = AStarVisualizer(array, path, search_steps, start, goal)
    plt.show()