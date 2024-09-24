# 本程序定义了一些辅助数据集生成的工具函数
# 作者：樊嘉森
# 最后修改时间：2024.9.24

import numpy as np
import subprocess


def apdl(mistune_pattern, freq):
    """
    函数功能：根据输入失谐模式及频率调用apdl计算真实响应
    mistune_pattern: 失谐模式列向量（N个扇区）
    freq: 频段列向量（S个频率点）
    返回 y_real: 真实响应矩阵（S行N列）
    """
    # 文件路径
    template_path = '../../data/apdl_template.txt'  # 模板文件路径
    ansys_executable = "E:\\ansys\\v231\\ansys\\bin\\winx64\\ANSYS231.exe"  # ANSYS可执行文件路径
    working_directory = "C:\\Users\\ljl\\Desktop\\test"  # 工作目录

    # 生成apdl文件
    EXX = 1.21E5  # 名义弹性模量
    exx_list = [(mistune_pattern[j] + 1) * EXX for j in range(12)]
    msg = '\n'.join(['*SET,EXX_{},{}'.format(j + 1, int(exx_list[j])) for j in range(12)])
    msg += '\n' + '*SET,W_1,{}'.format(int(np.min(freq))) + '\n' + '*SET,W_2,{}'.format(
        int(np.max(freq))) + '\n' + '*SET,K,{}'.format(
        len(freq))
    with open(template_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    lines.insert(10, msg)
    s = ''.join(lines)
    path = working_directory + '\\apdl.txt'
    with open(path, 'w') as file:
        file.write(s)

    # 调用ANSYS计算响应
    ansys_cmd = [ansys_executable, '-dir', working_directory, '-i', path]
    subprocess.run(ansys_cmd, cwd=working_directory, check=True)
    output_path = working_directory + '\\TIPDATA.txt'

    # 读取ANSYS计算结果
    data = np.loadtxt(output_path, dtype=np.float32)
    im, re = data[:, 1::2], data[:, ::2]
    y_real = np.sqrt(im ** 2 + re ** 2)

    return y_real


def AF_compute(y_real, y_tuned):
    """
    函数功能：计算响应放大因子
    y_real: 真实响应矩阵（N行S*Blade_num列）
    y_tuned: 调谐响应矩阵（N行S列）
    返回 AF_list: 响应放大因子列表（N行1列）
    """
    AF_list = np.zeros((y_real.shape[0], 1))
    for i in range(y_real.shape[0]):
        AF_list[i, 0] = np.max(y_real[i]) / np.max(y_tuned)

    return AF_list
