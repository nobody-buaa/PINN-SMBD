# 本程序定义了一些辅助模型训练的工具函数
# 作者：樊嘉森
# 最后修改时间：2024.9.24

import numpy as np
from sklearn.metrics import r2_score


def data_process(x, freq):
    """
    函数功能：将划分后的数据集均转换成规定格式 \n
    x: 输入数据集（N个样本，12个特征）\n
    freq: 频率列向量（S个频率点）\n
    返回 inputs: 规定格式的输入数据集（N*S行，13列）\n
    """
    num_samples = x.shape[0]
    num_freq = len(freq)
    inputs = np.zeros((num_samples * num_freq, 12 + 1))
    for i in range(num_samples):
        inputs[i * num_freq: (i + 1) * num_freq, :-1] = x[i, :]
        inputs[i * num_freq: (i + 1) * num_freq, -1] = freq.T
    return inputs


def score(y_true, y_predict):
    """
    函数功能：评价函数，计算R2值
    y_true: 真实值
    y_predict: 预测值
    返回 x: R2值(加权平均)
    """
    x = r2_score(y_true, y_predict, multioutput='uniform_average')
    return x


def data_augmentation(x, y):
    """
    函数功能：数据增强-移位特性
    x: 输入数据集（N*S行，13列）
    y: 输出数据集（N*S行，12列）
    返回 x_new: 增强后的输入数据集（N*S*12行，13列）
    返回 y_new: 增强后的输出数据集（N*S*12行，12列）
    """
    num_blade = x.shape[1] - 1
    x_new = np.zeros((x.shape[0] * num_blade, x.shape[1]))
    y_new = np.zeros((y.shape[0] * num_blade, y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(num_blade):
            x_new[i * num_blade + j, :-1] = np.roll(x[i, :-1], j)
            x_new[i * num_blade + j, -1] = x[i, -1]
            y_new[i * num_blade + j] = np.roll(y[i], j)
    print('Data augmentation is used!')
    return x_new, y_new
