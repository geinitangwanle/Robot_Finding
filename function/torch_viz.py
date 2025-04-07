import torch
from torchviz import make_dot
import torch.nn as nn

# 定义你自己的 DQN 网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc4(x)

# 创建一个模型实例
state_size = 900  # 这个根据你的 grid_size 来确定
action_size = 4  # 动作空间大小

model = DQN(state_size, action_size)

# 创建一个示例输入，来生成网络图
x = torch.ones(1, state_size)  # 假设一个 batch size 为 1 的输入，尺寸为 (1, state_size)

# 获取模型输出
y = model(x)

# 使用 torchviz 绘制模型图
dot = make_dot(y, params=dict(model.named_parameters()))

# 渲染并保存为图片
dot.render("DQN_network", format="png")
