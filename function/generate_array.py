import numpy as np
import os
from collections import deque

def generate_complex_scene(width, height, obstacle_density=0.2, num_islands=3, island_size_range=(3, 6)):
    # 初始化场景图
    scene = np.zeros((height, width))

    # 随机生成障碍物
    for _ in range(int(width * height * obstacle_density)):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        scene[y, x] = 1

    # 随机生成障碍物岛屿
    for _ in range(num_islands):
        island_size = np.random.randint(*island_size_range)
        start_x = np.random.randint(0, width - island_size)
        start_y = np.random.randint(0, height - island_size)
        for i in range(island_size):
            for j in range(island_size):
                scene[start_y + i, start_x + j] = 1

    # 验证场景图是否有通路
    def is_path_valid(scene):

        height, width = scene.shape
        visited = np.zeros_like(scene, dtype=bool)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 起始点和目标点
        start = (0, 0)
        end = (height - 1, width - 1)

        # 如果起始点或目标点是障碍物，直接返回 False
        if scene[start] == 1 or scene[end] == 1:
            return False

        # BFS 搜索
        queue = deque([start])
        visited[start] = True

        while queue:
            x, y = queue.popleft()
            if (x, y) == end:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny] and scene[nx, ny] == 0:
                    visited[nx, ny] = True
                    queue.append((nx, ny))

        return False

    # 确保生成的场景图有通路
    while not is_path_valid(scene):
        scene = generate_complex_scene(width, height, obstacle_density, num_islands, island_size_range)

    return scene

width = 30
height = 30
# 批量生成多个场景图并保存
num_scenes = 1  # 生成的场景图数量
output_dir = "scene_data/"

os.makedirs(output_dir, exist_ok=True)

for i in range(num_scenes):
        scene = generate_complex_scene(width, height)
        scene = scene.astype(int)
        file_path = os.path.join(output_dir, f"complex_scene_{i + 1}.txt")
        with open(file_path, "w") as f:
            for row in scene:
                f.write(" ".join(map(str, row)) + "\n")
