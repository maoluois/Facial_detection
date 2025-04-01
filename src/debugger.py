import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict

class DataDebugger:
    def __init__(self, history_length=100):
        """
        初始化调试器
        :param history_length: 历史数据最大长度
        """
        self.history_length = history_length
        self.data_history = defaultdict(list)  # 存储数据

        # 创建绘图窗口
        self.fig, self.ax = plt.subplots()
        self.ani = animation.FuncAnimation(self.fig, self._animate, interval=100)
        plt.show(block=False)

    def update(self, key, value):
        """
        更新数据并刷新绘图
        :param key: 数据名称（如 'yaw', 'speed', 'acceleration'）
        :param value: 当前值
        """
        self.data_history[key].append(value)

        # 仅保留最新 history_length 个数据
        if len(self.data_history[key]) > self.history_length:
            self.data_history[key].pop(0)

    def _animate(self, _):
        """定时刷新曲线"""
        self.ax.clear()
        self.ax.set_xlim(0, self.history_length)

        # 遍历所有数据流
        for key, values in self.data_history.items():
            self.ax.plot(range(len(values)), values, label=key)

        self.ax.legend()
        self.ax.set_title("Real-time Data Monitoring")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Value")


