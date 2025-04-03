import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

class AStarVisualizer:
    def __init__(self, array, path, search_steps, start, goal):
        self.array = array
        self.path = path
        self.search_steps = search_steps
        self.start = start
        self.goal = goal
        self.current_frame = 0
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # 定义颜色映射
        self.cmap = ListedColormap(['white', 'black', 'red', 'green', 'yellow', 'purple'])
        # 白色: 可通行, 黑色: 障碍物, 红色: 当前节点, 绿色: 起点/终点, 黄色: 开放列表, 紫色: 关闭列表
        
        # 准备所有帧
        self.prepare_frames()
        
        # 显示初始帧
        self.img = self.ax.imshow(self.frames[0], cmap=self.cmap)
        self.ax.set_title("A* Search Algorithm Visualization (Click to advance)")
        
        # 连接点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def prepare_frames(self):
        """准备所有动画帧"""
        self.frames = []
        
        # 添加搜索过程帧
        for step in self.search_steps:
            frame = np.zeros_like(self.array)
            frame[self.array == 1] = 1  # 障碍物
            
            # 开放列表 (黄色)
            for node in step['open_set']:
                frame[node[0], node[1]] = 4
            
            # 关闭列表 (紫色)
            for node in step['close_set']:
                frame[node[0], node[1]] = 5
            
            # 当前节点 (红色)
            frame[step['current'][0], step['current'][1]] = 2
            
            # 起点和终点 (绿色)
            frame[self.start[0], self.start[1]] = 3
            frame[self.goal[0], self.goal[1]] = 3
            
            self.frames.append(frame)
        
        # 添加路径帧
        if self.path:
            path_frame = np.zeros_like(self.array)
            path_frame[self.array == 1] = 1
            
            # 绘制路径 (红色)
            for node in self.path:
                path_frame[node[0], node[1]] = 2
            
            # 起点和终点 (绿色)
            path_frame[self.start[0], self.start[1]] = 3
            path_frame[self.goal[0], self.goal[1]] = 3
            
            # 添加多帧使路径显示更久
            self.frames.extend([path_frame] * 5)
    
    def on_click(self, event):
        """鼠标点击事件处理"""
        if event.inaxes != self.ax:
            return
        
        # 前进到下一帧
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.img.set_array(self.frames[self.current_frame])
        
        # 更新标题显示当前帧信息
        if self.current_frame < len(self.search_steps):
            step = self.search_steps[self.current_frame]
            info = f"Step {self.current_frame+1}/{len(self.search_steps)} - Current: {step['current']} - Open: {len(step['open_set'])} - Closed: {len(step['close_set'])}"
        else:
            info = "Path found! Click to restart from beginning."
        
        self.ax.set_title(f"A* Search Algorithm Visualization\n{info}")
        
        # 重绘图形
        self.fig.canvas.draw()

def visualize_search(array, path, search_steps, start, goal):
    """创建并显示交互式可视化"""
    visualizer = AStarVisualizer(array, path, search_steps, start, goal)
    plt.show()