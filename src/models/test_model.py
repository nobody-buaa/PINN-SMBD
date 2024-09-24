# 本程序用于模型的测试
# 作者：樊嘉森
# 最后修改时间：2024.9.24

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# 导入自定义模块
from src.models.model import DS, FC, DiabetesDataset
from src.utils.model_utils import data_process
from src.utils.test_utils import R2_compute, AF_compute, VLF_compute, max_freq, max_sector

# 常量定义
Freq = np.linspace(95, 115, 400)  # 频率范围


def statistical_analysis(set, mode, key, *data):
    """
    函数功能：失谐统计分析
    set: 模式（train/val/test/more_test/else）
    mode: 模型选择(PINN-SMBD,Without DS,Without Shift)
    key: 模型编号
    """
    # 常量定义
    train_num_list = [5, 10, 15, 20, 25, 30]

    # 训练数据导入
    x_train = np.loadtxt('../../data/train_set/x_train.txt', dtype=np.float32)
    y_train = np.loadtxt('../../data/train_set/y_real.txt', dtype=np.float32)
    freq_train = Freq

    # 选出实际训练数据
    x_train = x_train[:train_num_list[key]]
    y_train = y_train[:train_num_list[key] * len(freq_train)]

    # 测试数据导入
    if set == 'train' or set == 'val' or set == 'test' or set == 'more_test':
        x_test = np.loadtxt('../../data/{}_set/x_{}.txt'.format(set, set), dtype=np.float32)
        y_test_real = np.loadtxt('../../data/{}_set/y_real.txt'.format(set), dtype=np.float32)
        freq_test = Freq
    else:
        x_test = data[0]
        y_test_real = data[1]
        freq_test = data[2]

    # 数据预处理
    x_max = np.max(x_train)
    x_min = np.min(x_train)

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    freq = 2 * (freq_test - np.min(freq_train)) / (np.max(freq_train) - np.min(freq_train)) - 1  # 归一化

    # 数据预处理
    x_test = 2 * (x_test - x_min) / (x_max - x_min) - 1  # 归一化

    # 数据格式转换及转为tensor
    x_test = data_process(x_test, freq)

    # 数据加载器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = DiabetesDataset(x_test, y_test_real, device)
    batch_size = 4000  # 400的整数倍
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 根据模式选择模型
    if mode == 'Without DS':
        model = FC().to(device)
    elif mode == 'Without Shift':
        model = DS().to(device)
    elif mode == 'PINN-SMBD':
        model = DS().to(device)
    else:
        raise ValueError('Invalid mode!')
    model.load_state_dict(
        torch.load('../../models/weights/{}/best_weight_{}.pth'.format(mode, key)))

    # 模型预测
    y_test_predict = np.zeros_like(y_test_real)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            y_pred_test = model(inputs)
            y_pred_test = y_pred_test.cpu().numpy()
            y_test_predict[i * batch_size: (i + 1) * batch_size, :] = y_pred_test

    y_test_predict = y_test_predict * y_std + y_mean  # 反归一化

    # 格式转换
    y_test_real = y_test_real.reshape(-1, y_test_real.shape[1] * len(freq))
    y_test_predict = y_test_predict.reshape(-1, y_test_predict.shape[1] * len(freq))

    y_tuned = np.loadtxt('../../data/tuned/y.txt', dtype=np.float32).reshape(-1, y_test_real.shape[1])

    # 计算R2系数
    R2_list, total_R2 = R2_compute(y_test_predict, y_test_real, freq_test)
    # 计算响应放大因子
    AF_list, AF_error = AF_compute(y_test_predict, y_test_real, y_tuned)
    # 计算振动局部化因子
    VLF_list, VLF_error = VLF_compute(y_test_predict, y_test_real, freq_test)
    # 计算最大响应对应的频率
    max_freq_list, max_freq_error = max_freq(y_test_predict, y_test_real, freq_test)
    # 计算最大响应对应的扇区
    max_sector_list, max_sector_error = max_sector(y_test_predict, y_test_real, freq_test)

    # 打印结果
    print('Model: {}'.format(mode), 'Train_num: {}'.format(train_num_list[key]), 'set: {}'.format(set))
    print('R2 score: {:.6f}'.format(total_R2))
    print('AF_error: {:.4f}%'.format(AF_error))
    print('VLF_error: {:.4f}%'.format(VLF_error))
    print('max_freq_error: {:.4f}%'.format(max_freq_error))
    print('max_sector_error: {:.4f}%'.format(max_sector_error))

    return total_R2, AF_error, VLF_error, max_freq_error, max_sector_error, R2_list, max_freq_list


def test_num_analysis(mode, key):
    """
    函数功能：绘制不同测试集规模的模型失谐统计分析
    mode: 模型选择(PINN-SMBD,Without DS,Without Shift)
    key: 模型编号
    """
    # 读取扩充测试数据集
    test_num_list = [500, 750, 1000, 1250, 1500]
    x_more_test = np.loadtxt('../../data/more_test_set/x_more_test.txt', dtype=np.float32)
    y_more_test = np.loadtxt('../../data/more_test_set/y_real.txt', dtype=np.float32)
    freq_more_test = Freq

    # 随机打乱原始数据集
    ind = np.random.permutation(len(x_more_test))
    x_more_test = x_more_test[ind]
    y_more_test = y_more_test.reshape(-1, x_more_test.shape[1] * len(freq_more_test))[ind]

    # 将扩充测试数据集随机划分分为5个部分，每部分的样本数分别为500, 750, 1000, 1250, 1500，注意顺次不重复拿
    x_more_test_list = []
    y_more_test_list = []
    for i in range(len(test_num_list)):
        x_more_test_list.append(x_more_test[:test_num_list[i]])
        y_more_test_list.append(y_more_test[:test_num_list[i]].reshape(-1, x_more_test.shape[1]))
        x_more_test = x_more_test[test_num_list[i]:]
        y_more_test = y_more_test[test_num_list[i]:]

    # 统计分析
    test_score_list = []
    AF_error_list = []
    VLF_error_list = []
    max_freq_error_list = []
    max_sector_error_list = []

    for i in range(len(test_num_list)):
        total_R2, AF_error, VLF_error, max_freq_error, max_sector_error, _, _ = statistical_analysis('else', mode,
                                                                                                        key,
                                                                                                        x_more_test_list[
                                                                                                            i],
                                                                                                        y_more_test_list[
                                                                                                            i],
                                                                                                        freq_more_test)
        test_score_list.append(total_R2)
        AF_error_list.append(AF_error)
        VLF_error_list.append(VLF_error)
        max_freq_error_list.append(max_freq_error)
        max_sector_error_list.append(max_sector_error)

    # 绘制折线图
    plt.figure()
    plt.plot(test_num_list, max_sector_error_list, '-o', color='purple', label='Maximum-response blade')
    plt.plot(test_num_list, AF_error_list, '-o', color='b', label='Response amplification factor')
    plt.plot(test_num_list, VLF_error_list, '-o', color='g', label='Vibration localization factor')
    plt.plot(test_num_list, max_freq_error_list, '-o', color='orange', label='Maximum-response frequency')
    plt.xlabel('Testing set size')
    plt.ylabel('Error (%)')
    plt.title('Statistical analysis({}-{})'.format(mode, key))
    plt.legend()
    plt.xticks(test_num_list)
    plt.grid()
    plt.savefig('../../results/figures/test_num_analysis_{}_{}.png'.format(mode, key))
    plt.show()


if __name__ == '__main__':
    for i in range(6):
        test_num_analysis('PINN-SMBD', i)  # 某模型某权重在不同测试集规模上的失谐统计分析
