# 本程序用于可视化训练结果
# 作者：樊嘉森
# 最后修改时间：2024.9.24

import numpy as np
import matplotlib.pyplot as plt


def train_num_analysis(mode):
    """
    函数功能：绘制不同训练集规模的模型最终预测精度
    mode:模型选择(PINN-SMBD,Without DS,Without Shift)
    """
    train_num_list = [5, 10, 15, 20, 25, 30]
    best_r2 = []
    for i in range(len(train_num_list)):
        score_val = np.loadtxt('../../results/logs/{}/train_num_{}/score_val_{}.txt'.format(mode, i, i))
        score_test = np.loadtxt('../../results/logs/{}/train_num_{}/score_test_{}.txt'.format(mode, i, i))
        ind = np.argmax(score_val)
        best_r2.append(score_test[ind])
    # 保存best_r2
    np.savetxt('../../results/logs/{}/best_r2.txt'.format(mode), best_r2)


def model_analysis():
    """
    函数功能：绘制不同模型的最终预测精度
    """
    # 三种模型在不同训练集规模下在测试集上的预测R2系数
    train_num_analysis('PINN-SMBD')
    train_num_analysis('Without DS')
    train_num_analysis('Without Shift')

    train_num_list = [5, 10, 15, 20, 25, 30]
    model_list = ['PINN-SMBD', 'Without DS', 'Without Shift']
    best_r2 = np.zeros((len(model_list), len(train_num_list)))

    # 读取best_r2
    for i in range(len(model_list)):
        best_r2[i] = np.loadtxt('../../results/logs/{}/best_r2.txt'.format(model_list[i]))

    # 绘制折线图对比四种模型
    plt.figure()
    plt.plot(train_num_list, best_r2[0], '-o', label='PINN-SMBD')
    plt.plot(train_num_list, best_r2[2], '-o', label='Without DS')
    plt.plot(train_num_list, best_r2[1], '-o', label='Without Shift')
    plt.xlabel('Training set size')
    plt.ylabel('R2-score')
    plt.legend()
    plt.title('Comparison of different models')
    plt.savefig('../../results/figures/model_train_num_analysis.png')
    plt.show()


if __name__ == '__main__':
    # 三种模型在不同训练集规模上的预测精度的对比图
    model_analysis()

