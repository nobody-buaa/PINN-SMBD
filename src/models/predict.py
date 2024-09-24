# # 本程序用于预测模型
# # 作者：樊嘉森
# # 最后修改时间：2024.9.24

import numpy as np
import matplotlib.pyplot as plt

# 导入自定义模块
from src.utils.data_utils import apdl
from src.utils.model_utils import score
from src.utils.predict_utils import model_predict

# 定义常量
Freq = np.linspace(95, 115, 400)  # 频率范围


def real_predict(mistune_pattern, freq, mode, key, data, y_real=np.zeros((400, 12))):
    """
    函数功能：绘制真实和预测的频率-幅值图、频率-包络线图、频率-百分响应误差图
    mistune_pattern: 失谐模式
    freq: 频率列向量
    mode: 模型选择(PINN-SMBD,Without DS,Without Shift)
    key: 权重编号
    data: 数据集信息
    y_real: 真实响应矩阵
    """
    # 判断是否输入了y_real
    if np.sum(y_real) == 0:
        # 调用ANSYS计算响应
        y_real = apdl(mistune_pattern, freq)
    else:
        y_real = y_real

    # 计算预测值和R2
    y_predict = model_predict(mistune_pattern, freq, mode, key)
    pre_score = score(y_real, y_predict)

    # 谐调数据读取
    y_tuned = np.loadtxt('../../data/tuned/y.txt', dtype=np.float32)
    y_tuned = np.max(y_tuned, axis=1)
    freq_tuned = Freq

    # 计算响应放大因子
    AF_predict = np.max(y_predict) / np.max(y_tuned)
    AF_real = np.max(y_real) / np.max(y_tuned)

    # 绘制频率-幅值图
    plt.figure()
    # 创建一个颜色列表，共有12种颜色，不要重复,彼此区分度高
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink', 'purple', 'brown', 'gray']
    for i in range(y_predict.shape[1]):
        plt.plot(freq, y_real[:, i], color=color_list[i], linestyle='-')
        plt.scatter(freq, y_predict[:, i], color=color_list[i], marker='o', facecolors='none')
    plt.plot([], [], color='k', linestyle='-', label='Real')
    plt.scatter([], [], color='k', marker='o', facecolors='none', label='Predict')
    # 手动条件显示频率范围
    # plt.xlim(101, 105)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (mm)')
    plt.title('Real and Predicted Amplitude(R2={:.4f})'.format(pre_score))
    plt.grid()
    plt.legend()
    plt.savefig('../../results/predictions/real_predict_{}_{}_{}_{}.png'.format(mode, key, data[0], data[1]))

    # 绘制频率-包络线图
    plt.figure()
    # 对y_predict和y_real取每行的最大值
    y_predict_max = np.max(y_predict, axis=1)
    y_real_max = np.max(y_real, axis=1)
    plt.plot(freq, y_real_max, color='k', linestyle='-', label='Real(AF:{:.3f})'.format(AF_real))
    plt.plot(freq, y_predict_max, color='r', linestyle='--', label='Predict(AF:{:.3f})'.format(AF_predict))
    plt.fill_between(freq_tuned, y_tuned, 0, color='gray', alpha=0.5, label='Tuned')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (mm)')
    plt.title('Real and Predicted Envelope(R2={:.4f})'.format(pre_score))
    plt.grid()
    plt.legend()
    plt.savefig('../../results/predictions/envelope_predict_{}_{}_{}_{}.png'.format(mode, key, data[0], data[1]))

    # 绘制频率-百分响应误差图
    plt.figure()
    error = np.abs(y_real_max - y_predict_max) / np.max(y_real) * 100
    plt.plot(freq, error, color='r', linestyle='-', label='Error')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Error (%)')
    plt.title('Error of Envelope(R2={:.4f})'.format(pre_score))
    plt.grid()
    plt.legend()
    plt.savefig('../../results/predictions/error_predict_{}_{}_{}_{}.png'.format(mode, key, data[0], data[1]))

    plt.show()


def set_predict(set, mode, key, ind):
    """
    函数功能：数据集某一样本预测真实对比
    set: 模式（train/val/test）
    mode: 模型选择(PINN-SMBD,Without DS,Without Shift)
    key: 权重编号
    ind: 样本索引
    """
    # 选择观察的数据集中的样本
    x = np.loadtxt('../../data/{}_set/x_{}.txt'.format(set, set), dtype=np.float32)
    y = np.loadtxt('../../data/{}_set/y_real.txt'.format(set), dtype=np.float32)
    freq = Freq

    x = x[ind].reshape(1, -1)
    y_real = y[ind * len(freq):(ind + 1) * len(freq)]

    real_predict(x, freq, mode, key, [set, ind], y_real)


if __name__ == '__main__':
    ind_list = [0, 1, 6, 7, 212]  # 测试集中的观察样本索引
    for i in range(len(ind_list)):
        # set_predict('test', 'PINN-SMBD', 3, ind_list[i])  # 显示频率范围手动调节
        pass
