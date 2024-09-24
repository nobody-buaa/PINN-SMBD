# 本程序定义了一些辅助模型测试的工具函数
# 作者：樊嘉森
# 最后修改时间：2024.9.24

import numpy as np
from sklearn.metrics import r2_score as score


def R2_compute(y_predict, y_real, freq):
    """
    函数功能：计算R2值
    y_predict: 预测响应矩阵（N行S*Blade_num列）
    y_real: 真实响应矩阵（N行S*Blade_num列）
    freq: 频率列向量（S个频率点）
    返回 R2_list: R2值列表（N行1列）
    返回 total_R2: 整体R2值
    """
    sample_num = y_predict.shape[0]
    R2_list = np.zeros(sample_num)
    for i in range(sample_num):
        temp_pred = y_predict[i].reshape(len(freq), -1)
        temp_real = y_real[i].reshape(len(freq), -1)
        R2_list[i] = score(temp_real, temp_pred)

    total_R2 = score(y_real.reshape(sample_num * len(freq), -1), y_predict.reshape(sample_num * len(freq), -1))

    return R2_list, total_R2


def AF_compute(y_predict, y_real, y_tuned):
    """
    函数功能：计算响应放大因子
    y_predict: 预测响应矩阵（N行S*Blade_num列）
    y_real: 真实响应矩阵（N行S*Blade_num列）
    y_tuned: 调谐响应矩阵（N行S列）
    返回 AF_list: 响应放大因子列表（N行2列）
    返回 AF_error: 响应放大因子相对误差均值
    """
    AF_list = np.zeros((y_predict.shape[0], 2))
    for i in range(y_predict.shape[0]):
        AF_list[i, 0] = np.max(y_predict[i]) / np.max(y_tuned)
        AF_list[i, 1] = np.max(y_real[i]) / np.max(y_tuned)
    # 计算AF的相对误差的均值
    AF_error = np.mean(np.abs(AF_list[:, 0] - AF_list[:, 1]) / AF_list[:, 1]) * 100

    return AF_list, AF_error


def VLF_compute(y_predict, y_real, freq):
    """
    函数功能：计算振动局部化因子
    y_predict: 预测响应矩阵（N行S*Blade_num列）
    y_real: 真实响应矩阵（N行S*Blade_num列）
    freq: 频率列向量（S个频率点）
    返回 VLF_list: 振动局部化因子列表（N行2列）
    返回 VLF_error: 振动局部化因子相对误差均值
    """
    VLF_list = np.zeros((y_predict.shape[0], 2))
    for i in range(y_predict.shape[0]):
        temp_pred = y_predict[i].reshape(len(freq), -1)
        temp_real = y_real[i].reshape(len(freq), -1)
        VLF_list[i, 0] = np.max(temp_pred) / np.mean(np.max(temp_pred, axis=0))
        VLF_list[i, 1] = np.max(temp_real) / np.mean(np.max(temp_real, axis=0))
    # 计算VLF的相对误差的均值
    VLF_error = np.mean(np.abs(VLF_list[:, 0] - VLF_list[:, 1]) / VLF_list[:, 1]) * 100

    return VLF_list, VLF_error


def max_freq(y_predict, y_real, freq):
    """
    函数功能：计算最大响应对应的频率
    y_predict: 预测响应矩阵（N行S*Blade_num列）
    y_real: 真实响应矩阵（N行S*Blade_num列）
    freq: 频率列向量（S个频率点）
    返回 max_freq_list: 最大响应对应的频率列表（N行2列）
    返回 max_freq_error: 最大响应对应的频率相对误差均值
    """
    max_freq_list = np.zeros((y_predict.shape[0], 2))
    for i in range(y_predict.shape[0]):
        temp_pred = y_predict[i].reshape(len(freq), -1)
        max_index1 = np.argmax(np.max(temp_pred, axis=1))
        max_freq_list[i, 0] = freq[max_index1]

        temp_real = y_real[i].reshape(len(freq), -1)
        max_index2 = np.argmax(np.max(temp_real, axis=1))
        max_freq_list[i, 1] = freq[max_index2]
    # 计算最大响应对应的频率的相对误差的均值
    max_freq_error = np.mean(np.abs(max_freq_list[:, 0] - max_freq_list[:, 1]) / max_freq_list[:, 1]) * 100

    return max_freq_list, max_freq_error


def max_sector(y_predict, y_real, freq):
    """
    函数功能：计算最大响应对应的扇区
    y_predict: 预测响应矩阵（N行S*Blade_num列）
    y_real: 真实响应矩阵（N行S*Blade_num列）
    freq: 频率列向量（S个频率点）
    返回 max_sector_list: 最大响应对应的扇区列表（N行2列）
    返回 max_sector_error: 最大响应对应的扇区相对误差均值
    """

    max_sector_list = np.zeros((y_predict.shape[0], 2))
    for i in range(y_predict.shape[0]):
        temp_pred = y_predict[i].reshape(len(freq), -1)
        max_index1 = np.argmax(np.max(temp_pred, axis=0))
        max_sector_list[i, 0] = max_index1

        temp_real = y_real[i].reshape(len(freq), -1)
        max_index2 = np.argmax(np.max(temp_real, axis=0))
        max_sector_list[i, 1] = max_index2
    # 计算最大响应对应的扇区的正确率，即为两者相等的比例
    max_sector_acc = np.sum(max_sector_list[:, 0] == max_sector_list[:, 1]) / len(max_sector_list)
    max_sector_error = (1 - max_sector_acc) * 100

    return max_sector_list, max_sector_error
