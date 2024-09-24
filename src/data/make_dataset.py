# 本程序用于构造训练集、验证集和测试集
# 作者：樊嘉森
# 最后修改时间：2024.9.24

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import os

from src.utils.data_utils import apdl, AF_compute

Blade_num = 12  # 叶片数
Freq = np.linspace(95, 115, 400)  # 频率范围


def normal_distribution(n, deta=0.2):
    """
    函数功能：多元正态分布探究，在分布稳定时确定高方差范围
    n : 样本数
    deta : 标准差
    """
    x = np.random.normal(0, 1, (n, Blade_num)) * deta
    var = np.var(x, axis=1)

    plt.figure()
    plt.hist(var, bins=20, color='steelblue', edgecolor='k')
    plt.xlabel('variance')
    plt.ylabel('frequency')
    plt.title('variance distribution (n={})'.format(n))
    plt.show()


def train_set_construct(train_num=30, deta=0.2, arrange=np.array([0.06, 0.08])):
    """
    函数功能：训练集构造,并绘制方差折线图,保存训练集为txt文件
    train_num : 训练集样本数
    deta : 标准差
    arrange : 方差范围，已给出默认值
    """
    x = np.random.normal(0, 1, (10000, Blade_num)) * deta
    var = np.var(x, axis=1)
    ind = np.argsort(var)[::-1]  # 按照var从大到小的顺序排序
    x = x[ind]
    var = var[ind]

    # 在方差范围为arrange的样本中随机选择train_num个样本
    ind1 = np.where((var >= arrange[0]) & (var <= arrange[1]))[0]
    x = x[ind1]

    ind2 = np.random.choice(x.shape[0], train_num, replace=False)
    x_train = x[ind2]
    var_train = np.var(x_train, axis=1)

    # 按照方差从大到小的顺序排序
    ind3 = np.argsort(var_train)[::-1]
    x_train = x_train[ind3]
    var_train = var_train[ind3]

    # 绘制方差折线图
    plt.figure()
    plt.plot(range(1, train_num + 1), var_train, marker='o', color='steelblue')
    plt.xlabel('Sample number')
    plt.ylabel('Variance')
    plt.title('Variance line chart(train)')
    plt.grid()
    plt.savefig('../../data/train_set/variance_line_chart_train.png')
    plt.show()

    # 保存为txt文件
    np.savetxt('../../data/train_set/x_train.txt', x_train, fmt='%.4f', delimiter=' ')


def valortest_set_construct(mode, num=10, deta=0.2, nor_rate=0.5, lhs_rate=0.5, arrange=np.array([0, 2])):
    """
    函数功能：验证或测试集构造,并绘制方差折线图,保存为txt文件
    mode : 模式（train/val/test/more_test）
    num : 样本数
    deta : 标准差
    nor_rate : 正态分布样本比例
    lhs_rate : LHS样本比例
    arrange : 方差范围，已给出默认值
    """
    x = np.random.normal(0, 1, (10000, Blade_num)) * deta
    var = np.var(x, axis=1)
    ind = np.argsort(var)[::-1]  # 按照var从大到小的顺序排序
    x = x[ind]

    # 在x中的选择
    ind1 = np.where((var >= arrange[0]) & (var <= arrange[1]))[0]
    x = x[ind1]
    ind2 = np.random.choice(x.shape[0], int(num * nor_rate), replace=False)
    x1 = x[ind2]

    # 使用拉丁超立方采样生成样本，每个样本12个变量，每个变量的范围为[-0.3, 0.3]
    lower_bound = np.ones(12) * -0.3
    upper_bound = np.ones(12) * 0.3
    x2 = lhs(12, int(num * lhs_rate), criterion='maximin')
    x2 = x2 * (upper_bound - lower_bound) + lower_bound

    x_valortest = np.vstack((x1, x2))
    var_valortest = np.var(x_valortest, axis=1)

    # 按照方差从大到小的顺序排序
    ind3 = np.argsort(var_valortest)[::-1]
    x_valortest = x_valortest[ind3]
    var_valortest = var_valortest[ind3]

    # 绘制方差折线图
    plt.figure()
    plt.plot(range(1, num + 1), var_valortest, marker='o', color='steelblue')
    plt.xlabel('Sample number')
    plt.ylabel('Variance')
    plt.title('Variance line chart({})'.format(mode))
    plt.grid()
    plt.savefig('../../data/{}_set/variance_line_chart_{}.png'.format(mode, mode))
    plt.show()

    # 保存为txt文件
    np.savetxt('../../data/{}_set/x_{}.txt'.format(mode, mode), x_valortest, fmt='%.4f', delimiter=' ')


def apdl_compute(mode):
    """
    函数功能：调用apdl分别计算训练集、验证集和测试集的真实响应
    mode : 模式（train/val/test/more_test）
    """
    # 读取数据
    x = np.loadtxt('../../data/{}_set/x_{}.txt'.format(mode, mode), delimiter=' ')

    # 判断是否已经计算过真实响应
    if os.path.exists('../../data/{}_set/y_real.txt'.format(mode)):
        y_real = np.loadtxt('../../data/{}_set/y_real.txt'.format(mode), delimiter=' ')
        ind = np.where(np.sum(y_real, axis=1) == 0)[0]
        num = int(ind[0] / len(Freq))
        print('The number of samples that have been calculated:', num)
        y_real = np.vstack((y_real, np.zeros((x.shape[0] * len(Freq) - y_real.shape[0], Blade_num))))

    else:
        num = 0
        y_real = np.zeros((x.shape[0] * len(Freq), Blade_num))

    # 计算真实响应
    for i in range(num, x.shape[0]):
        print('\r{}/{}'.format(i + 1, x.shape[0]), end='')
        y_real[i * len(Freq): (i + 1) * len(Freq), :] = apdl(x[i], Freq)
        np.savetxt('../../data/{}_set/y_real.txt'.format(mode), y_real, fmt='%.4f', delimiter=' ')

    plot_AF(mode=mode)


def plot_AF(mode):
    """
    函数功能：绘制每个数据集的AF-样本序号图
    mode : 模式（train/val/test/more_test）
    """
    # 读取数据
    y_real = np.loadtxt('../../data/{}_set/y_real.txt'.format(mode), delimiter=' ')
    y_tuned = np.loadtxt('../../data/tuned/y.txt', delimiter=' ')

    # 计算AF
    AF_list = AF_compute(y_real.reshape(-1, Blade_num * len(Freq)), y_tuned)

    # 绘图
    plt.figure()
    plt.plot(range(1, AF_list.shape[0] + 1), AF_list[:, 0], color='r', marker='o')
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude Factor')
    plt.title('Amplitude Factor({})'.format(mode))
    plt.grid()
    plt.savefig('../../data/{}_set/AF_{}.png'.format(mode, mode))
    plt.show()


def tuned():
    """
    函数功能：绘制谐调时的频率-幅值图
    """
    # 计算数据
    x = np.zeros(Blade_num)
    y = apdl(x, Freq)

    # 绘图
    plt.figure()
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink', 'purple', 'brown', 'gray']
    for i in range(y.shape[1]):
        plt.plot(Freq, y[:, i], label='Blade {}'.format(i + 1), color=color_list[i], linestyle='-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (mm)')
    plt.title('Frequency-Amplitude Curve of Tuned Blisk')
    plt.legend()
    plt.grid()
    plt.savefig('../../data/tuned/Freq_amp.png')
    plt.show()

    np.savetxt('../../data/tuned/y.txt', y, fmt='%.4f', delimiter=' ')


if __name__ == '__main__':
    # -------------------------------多元正态分布探究，以确定高方差范围--------------------------------
    num_list = [100, 1000, 10000, 100000, 1000000]
    for num in num_list:
        # normal_distribution(num)
        pass

    # -------------------------------谐调叶盘计算--------------------------------------------------
    # tuned()

    # -------------------------------数据集构造-----------------------------------------------------
    # train_set_construct()  # 训练集构造
    # valortest_set_construct(mode='val', num=10)  # 验证集构造
    # valortest_set_construct(mode='test', num=500)  # 测试集构造
    # valortest_set_construct(mode='more_test', num=5000)  # 更多测试集构造

    # -------------------------------计算真实响应---------------------------------------------------
    # apdl_compute(mode='train')
    # apdl_compute(mode='val')
    # apdl_compute(mode='test')
    # apdl_compute(mode='more_test')
