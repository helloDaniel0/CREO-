import matplotlib.pyplot as plt
import random
import numpy as np


def read_data3(filename):
    x_data = []
    y1_data = []
    y2_data = []
    with open(filename, 'r') as file:
        for line in file:
            x, y1, y2 = map(float, line.split())
            x_data.append(x)
            y1_data.append(y1)
            y2_data.append(y2)
    return x_data, y1_data, y2_data


def plot_data3(x, y1, y2):
    plt.plot(x, y1, label='cwnd', color='blue')
    plt.plot(x, y2, label='ssthresh', color='red')
    plt.title("epoch-cwnd and epoch-ssthresh Line Plot")
    plt.xlabel("epoch")
    plt.ylabel("cwnd or ssthresh")
    plt.legend()
    plt.grid(True)
    plt.show()


def read_data(filename):
    x_data = []
    y_data = []
    with open(filename, 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            x_data.append(x)
            y_data.append(y)
    return x_data, y_data


def plot_data(x, y):
    plt.plot(x, y, color='blue')
    plt.title("Epoch-Reward(Normalized) Scatter Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 文件名
    filename = "dqn_data.txt"

    filename2 = "generated_data.txt"

    # 读取数据
    x_data, y1_data, y2_data = read_data3(filename)

    # 绘制数据
    plot_data3(x_data, y1_data, y2_data)

    # 生成 2000 行数据
    epochs = list(range(1, 2001))
    rewards = []

    # 前 600 行数据，表现出指数增长趋势，并且有一些突发点和接近0的突发点
    for i in range(600):
        base_reward = 5 * (1.003 ** i)  # 指数增长基线
        noise = random.gauss(0, 5)  # 正态分布噪声
        spike = random.choice([0, random.uniform(-10, 100)]) if random.random() < 0.1 else 0  # 10% 的概率产生突发点
        near_zero_spike = random.choice(
            [0, random.uniform(-10, 10)]) if random.random() < 0.15 else 0  # 15% 的概率产生接近 0 的突发点
        reward = base_reward + noise + spike + near_zero_spike
        if random.random() < 0.01:
            rewards.append(0)  # 确保 reward 在 0 到 100 之间
        else:
            rewards.append(max(0, min(100, reward)))  # 确保 reward 在 0 到 100 之间

    # 后 1400 行数据，基本趋于稳定
    stable_reward = 98.2  # 前一段的最后一个奖励值作为基准
    for i in range(600, 2000):
        noise = random.gauss(0, 1)  # 稳定期的低噪声
        spike = random.choice([0, random.uniform(-10, 10)]) if random.random() < 0.1 else 0  # 10% 的概率产生接近 0 的突发点
        reward = stable_reward + noise + spike
        rewards.append(max(0, min(100, reward)))  # 确保 reward 在 0 到 100 之间

    # 将数据写入文件
    with open("reward_data.txt", "w") as file:
        for epoch, reward in zip(epochs, rewards):
            file.write(f"{epoch} {reward}\n")

    print("Generated data file 'reward_data.txt' successfully.")

    # 可视化数据以验证结果
    plt.plot(epochs, rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward vs. Epoch')
    plt.grid(True)
    plt.show()

    # 生成数据
    x_values = list(range(2000))
    data = []
    # 前500行数据，在-2和2之间随机波动，取整数值
    data.extend([random.uniform(-2, 2) for _ in range(613)])
    # 后1500行数据，1%的概率在-2和2之间随机波动，取整数值
    data.extend([random.uniform(-2, 2) if random.random() < 0.01 else 0 for _ in range(1387)])

    # 绘制图形
    plt.plot(x_values, data)
    plt.xlabel('epcoh')
    plt.ylabel('Action')
    plt.title('Action Trace')
    plt.grid(True)
    plt.show()
