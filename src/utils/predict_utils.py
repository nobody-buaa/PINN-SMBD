# 本程序定义了一些辅助模型预测的工具函数
# 作者：樊嘉森
# 最后修改时间：2024.9.24

import numpy as np
import torch

# 导入自定义模块
from src.models.model import DS, FC
from src.utils.model_utils import data_process

# 定义常量
Freq = np.linspace(95, 115, 400)  # 频率范围


def model_predict(mistune_pattern, freq, mode, key):
    """
    函数功能：根据输入失谐模式及频率使用相应模型预测响应
    mistune_pattern: 失谐模式列向量（N个扇区）
    freq: 频率值(S个频率点）
    mode: 模型选择(PINN-SMBD,Without DS,Without Shift)
    key: 权重编号
    返回 y_predict: 预测响应矩阵（S行N列）
    """
    # 常量定义
    train_num_list = [5, 10, 15, 20, 25, 30]

    # 训练数据导入
    x_train = np.loadtxt('../../data/train_set/x_train.txt', dtype=np.float32)
    y_train = np.loadtxt('../../data/train_set/y_real.txt', dtype=np.float32)
    Freq_train = Freq

    # 选出实际训练数据
    x_train = x_train[:train_num_list[key]]
    y_train = y_train[:train_num_list[key] * len(Freq_train)]

    # 数据预处理
    x_max = np.max(x_train)
    x_min = np.min(x_train)

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    x = 2 * (mistune_pattern - x_min) / (x_max - x_min) - 1  # 归一化
    freq = 2 * (freq - np.min(Freq_train)) / (np.max(Freq_train) - np.min(Freq_train)) - 1  # 归一化

    # 数据格式转换及转为tensor
    x = data_process(x.reshape(1, -1), freq)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # 根据mode选择模型
    if mode == 'PINN-SMBD':
        model = DS()
    elif mode == 'Without DS':
        model = FC()
    elif mode == 'Without Shift':
        model = DS()
    else:
        raise ValueError('mode error')

    model.load_state_dict(
        torch.load('../../models/weights/{}/best_weight_{}.pth'.format(mode, key)))

    # 模型预测
    model.eval()
    y_predict = model(x_tensor).detach().numpy()
    y_predict = y_predict * y_std + y_mean  # 反归一化

    return y_predict


