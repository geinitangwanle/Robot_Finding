import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9):
        self.state_size = state_size #表示环境中状态的数量，用于确定 Q 表的行数。
        self.action_size = action_size #表示智能体可以采取的动作数量，用于确定 Q 表的列数。
        self.learning_rate = learning_rate #学习率，控制 Q 表更新的步长，默认值为 0.1。学习率越大，每次更新时 Q 值的变化就越大；学习率越小，更新就越缓慢。
        self.discount_factor = discount_factor #折扣因子，用于权衡当前奖励和未来奖励的重要性，默认值为 0.9。折扣因子越接近 1，智能体越重视未来的奖励；越接近 0，越重视当前的奖励。
        self.q_table = np.zeros((state_size, action_size)) #使用 numpy 的 zeros 函数创建一个形状为 (state_size, action_size) 的零矩阵，用于存储每个状态 - 动作对的 Q 值。

    def choose_action(self, state, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])     
    '''
    state：当前智能体所处的状态。
    epsilon：探索率，默认值为 0.1。用于平衡探索（随机选择动作）和利用（选择 Q 值最大的动作）。
    当 np.random.uniform(0, 1) < epsilon 时，以 epsilon 的概率进行探索，随机选择一个动作。
    否则，以 1 - epsilon 的概率进行利用，选择当前状态下 Q 值最大的动作。
    '''

    def update_q_table(self, state, action, reward, next_state):
        max_q_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * max_q_next - self.q_table[state, action])
