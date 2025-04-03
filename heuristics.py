def heuristic(a, b):
    # 曼哈顿距离作为启发式函数
    return abs(a[0] - b[0]) + abs(a[1] - b[1])