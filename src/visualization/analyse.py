# 本程序用于模型预测结果的可视化分析
# 作者：樊嘉森
# 最后修改时间：2024.9.24

import numpy as np
import matplotlib.pyplot as plt

# 导入自定义模块
from src.models.test_model import statistical_analysis


def calculation(mode, key):
    """
    首先计算模型的在扩充数据集上的预测精度、频率偏差
    mode: 模型选择(PINN-SMBD,Without DS,Without Shift)
    key: 模型编号
    """
    _, _, _, _, _, R2_list, max_freq_list = statistical_analysis('more_test', mode, key)
    # 保存数据
    np.savetxt('../../results/analysis/R2_list_{}_{}_more_test.txt'.format(mode, key), R2_list)
    np.savetxt('../../results/analysis/max_freq_list_{}_{}_more_test.txt'.format(mode, key), max_freq_list)


def analyse(mode, key):
    # 加载数据
    R2_list = np.loadtxt('../../results/analysis/R2_list_{}_{}_more_test.txt'.format(mode, key))
    max_freq_list = np.loadtxt('../../results/analysis/max_freq_list_{}_{}_more_test.txt'.format(mode, key))
    mistune_pattern = np.loadtxt('../../data/more_test_set/x_more_test.txt')

    # 分别统计R2值在<0.98,0.98-0.99,>0.99的数目
    R2_1 = np.sum(R2_list < 0.98)
    R2_2 = np.sum((R2_list >= 0.98) & (R2_list < 0.99))
    R2_3 = np.sum(R2_list >= 0.99)

    # 绘制R2值的饼图
    plt.figure()
    labels = ['R2<0.98', '0.98<=R2<0.99', 'R2>=0.99']
    sizes = [R2_1, R2_2, R2_3]
    explode = (0.3, 0, 0)
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%', startangle=-180 * sizes[0])
    plt.axis('equal')
    plt.title('R2 Value Distribution')
    plt.savefig('../../results/analysis/R2_distribution_{}_{}.png'.format(mode, key))
    plt.show()

    # 在R2<0.98的数据中，统计频率偏差大于均值的数目
    tune_freq = 103.4094754302
    freq_deviation = np.abs(max_freq_list[R2_list < 0.98, 1] - tune_freq)
    mean_deviation = np.mean(np.abs(max_freq_list[:, 1] - tune_freq))
    dev_1 = np.sum(freq_deviation < mean_deviation)
    dev_2 = np.sum(freq_deviation >= mean_deviation)

    # 绘制频率分布的条形饼图
    categories = ['Under {:.2f}Hz'.format(mean_deviation), ' Over {:.2f}Hz'.format(mean_deviation)]  # 标签
    values = [dev_1 / (dev_1 + dev_2), dev_2 / (dev_1 + dev_2)]  # 数
    bottom = 1
    width = .2
    fig, ax = plt.subplots(figsize=(5, 7))
    for j, (height, label) in enumerate(reversed([*zip(values, categories)])):
        bottom -= height
        bc = ax.bar(0, height, width, bottom=bottom, color='C0', label=label,
                    alpha=0.7 - 0.4 * j)
        ax.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

    ax.set_title('Frequency deviation')
    ax.legend(loc=2, fontsize=9)
    ax.axis('off')
    ax.set_xlim(- 2.5 * width, 2.5 * width)
    plt.tight_layout()
    plt.savefig('../../results/analysis/freq_deviation_{}_{}.png'.format(mode, key))
    plt.show()

    # 在R2<0.98的数据中，统计失谐模式方差在<0.02,0.02-0.04,0.04-0.06,>0.06的数目
    variances = np.var(mistune_pattern[R2_list < 0.98], axis=1)
    var_1 = np.sum(variances < 0.02)
    var_2 = np.sum((variances >= 0.02) & (variances < 0.04))
    var_3 = np.sum((variances >= 0.04) & (variances < 0.06))
    var_4 = np.sum(variances >= 0.06)

    # 绘制失谐模式方差的饼图
    plt.figure()
    labels = ['>0.06', '0.04-0.06', '0.02-0.04', '<0.02']
    sizes = [var_4, var_3, var_2, var_1]
    colors = ['royalblue', 'cornflowerblue', 'lightsteelblue', 'slategrey']
    explode = [0.1, 0.1, 0, 0]
    plt.pie(sizes, labels=labels, autopct='%1.0f%%', shadow=True, startangle=90, colors=colors, explode=explode)
    plt.axis('equal')
    plt.title('Variance Distribution')
    plt.savefig('../../results/analysis/variance_distribution_{}_{}.png'.format(mode, key))
    plt.show()


if __name__ == '__main__':
    # calculation('PINN-SMBD', 3)
    # analyse('PINN-SMBD', 3)
    pass
