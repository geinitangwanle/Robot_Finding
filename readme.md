# Robot Finding 项目说明

## 项目概述
本项目主要围绕机器人路径规划展开，结合了 QNN（强化神经网络）、A* 算法等多种技术，实现了地图读取、路径搜索、模型训练以及搜索过程可视化等功能。

## 项目结构
- `Robot_Finding`
  - `QNN_train.py`：包含地图加载函数，用于 QNN 训练的前期准备。
  - `QNNAstar_train.py`：结合 QNN 和 A* 算法进行训练，使用地图数据进行模型训练。
  - `QNN_evaluate.py`：包含地图加载函数，用于 QNN 模型的评估。
  - `Q_learn_train_search_visualization.py`：包含地图加载函数，用于 Q 学习训练及搜索过程可视化。
  - `A*Search_visualization_auto.py`：使用 A* 算法进行路径搜索，并可视化搜索过程。
  - `function`：存放可视化相关的函数。
  - `scene_data`：存放各种地图场景的文本文件。
  - `scene`：存放自定义地图场景的文本文件。

## 运行步骤
1. 确保你已经安装了所需的依赖库，如 `numpy`、`torch`、`matplotlib` 等。
2. 可以根据需要修改代码中的地图文件路径和起点、终点坐标。
3. 运行相应的 Python 文件，如 `A*Search_visualization_auto.py` 来查看 A* 算法的搜索过程可视化。

## 注意事项
- 地图文件的格式应为文本文件，每行用空格分隔的数字表示地图的一行。
- 代码中的部分函数可能需要根据具体需求进行调整和优化。