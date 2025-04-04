import numpy as np

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

    return scene

# 示例：生成一个 20x20 的复杂场景图
width = 20
height = 20
scene = generate_complex_scene(width, height)
print(scene)
scene = scene.astype(int)

# 将生成的场景图保存为文本文件
with open("complex_scene.txt", "w") as f:
    for row in scene:
        f.write(" ".join(map(str, row)) + "\n")
