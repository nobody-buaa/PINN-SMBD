# 本程序用于训练模型
# 作者：樊嘉森
# 最后修改时间：2024.9.24

# 导入必要的库
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim

# 导入自定义模块
from model import FC, DS, DiabetesDataset
from src.utils.model_utils import data_process, score, data_augmentation

# 常量定义
Blade_num = 12
Freq = np.linspace(95, 115, 400)  # 频率范围


def train(mode, train_num, lr, key, weight, eval_epoch=1, epoch_num=1000):
    """
    函数功能：训练模型
    mode: 模型选择(PINN-SMBD,Without DS,Without Shift)
    train_num: 训练样本数量
    lr: 学习率
    key: 训练样本数量索引
    weight: 正则化系数
    eval_epoch: 评估间隔
    epoch_num: 训练轮数
    """
    # 导入数据
    x_train = np.loadtxt('../../data/train_set/x_train.txt', dtype=np.float32)
    y_train = np.loadtxt('../../data/train_set/y_real.txt', dtype=np.float32)
    freq_train = Freq

    x_val = np.loadtxt('../../data/val_set/x_val.txt', dtype=np.float32)
    y_val = np.loadtxt('../../data/val_set/y_real.txt', dtype=np.float32)
    freq_val = Freq

    x_test = np.loadtxt('../../data/test_set/x_test.txt', dtype=np.float32)
    y_test = np.loadtxt('../../data/test_set/y_real.txt', dtype=np.float32)
    freq_test = Freq

    # 选择前train_num个样本进行训练
    x_train = x_train[:train_num]
    y_train = y_train[:train_num * len(freq_train)]

    # 数据预处理
    x_max = np.max(x_train)
    x_min = np.min(x_train)

    x_train = 2 * (x_train - x_min) / (x_max - x_min) - 1
    x_val = 2 * (x_val - x_min) / (x_max - x_min) - 1
    x_test = 2 * (x_test - x_min) / (x_max - x_min) - 1

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    freq_val = 2 * (freq_val - np.min(freq_train)) / (np.max(freq_train) - np.min(freq_train)) - 1
    freq_test = 2 * (freq_test - np.min(freq_train)) / (np.max(freq_train) - np.min(freq_train)) - 1
    freq_train = 2 * (freq_train - np.min(freq_train)) / (np.max(freq_train) - np.min(freq_train)) - 1

    # 数据格式转换及增强
    x_train = data_process(x_train, freq_train)
    x_val = data_process(x_val, freq_val)
    x_test = data_process(x_test, freq_test)

    # 根据模式选择是否进行数据增强
    if mode == 'Without Shift':
        pass
    else:
        x_train, y_train = data_augmentation(x_train, y_train)

    # 训练设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = DiabetesDataset(x_train, y_train, device)
    val_dataset = DiabetesDataset(x_val, y_val, device)
    test_dataset = DiabetesDataset(x_test, y_test, device)

    train_dataloader = DataLoader(train_dataset, batch_size=2000, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(val_dataset, batch_size=4000, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=4000, shuffle=False, num_workers=0)

    # 根据模式选择模型
    if mode == 'Without DS':
        model = FC().to(device)
    elif mode == 'Without Shift':
        model = DS().to(device)
    elif mode == 'PINN-SMBD':
        model = DS().to(device)
    else:
        raise ValueError('Invalid mode!')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight)

    loss_list = []
    loss_val_list = []
    score_val_list = []
    score_test_list = []
    best_epoch, best_train_loss, best_val_score, best_test_score = 0, 0, 0, 0

    for epoch in range(epoch_num):
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            outputs = model(inputs)

            loss = criterion(outputs.float(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        loss_list.append(train_loss)

        if (epoch + 1) % eval_epoch == 0:
            print("epoch {}".format(epoch + 1) + " train loss :", train_loss, end=' , ')
            model.eval()
            with torch.no_grad():
                # 验证集评估
                val_score_sum = 0
                val_loss_sum = 0
                for x_val_batch, y_val_batch in val_dataloader:
                    y_pred_val = model(x_val_batch.to(device))
                    loss = criterion(y_pred_val, y_val_batch.to(device))
                    val_loss_sum += loss.item()
                    y_pred_val = y_pred_val.cpu().numpy()
                    score_val_batch = score(y_val_batch.cpu().numpy(), y_pred_val)
                    val_score_sum += score_val_batch
                avg_score_val = val_score_sum / len(val_dataloader)
                avg_loss_val = val_loss_sum / len(val_dataloader)
                score_val_list.append(avg_score_val)
                loss_val_list.append(avg_loss_val)
                print("score in val:", avg_score_val, end=' , ')

                # 测试集评估，过程类似验证集
                test_score_sum = 0
                for x_test_batch, y_test_batch in test_dataloader:
                    y_pred_test = model(x_test_batch.to(device))
                    y_pred_test = y_pred_test.cpu().numpy()
                    score_test_batch = score(y_test_batch.cpu().numpy(), y_pred_test)
                    test_score_sum += score_test_batch
                avg_score_test = test_score_sum / len(test_dataloader)
                score_test_list.append(avg_score_test)
                print("score in test:", avg_score_test, '\t')

            if avg_score_val == max(score_val_list):
                best_epoch = epoch + 1

                best_train_loss = train_loss
                best_val_score = avg_score_val
                best_test_score = avg_score_test

                torch.save(model.state_dict(), '../../models/weights/{}/best_model_{}.pth'.format(mode, key))

    print('best epoch:', best_epoch, end=' , ')
    print('best loss in train:', best_train_loss, end=' , ')
    print('best score in val:', best_val_score, end=' , ')
    print('best score in test:', best_test_score, '\t')

    # 保存训练过程中的数据
    np.savetxt('../../results/logs/{}/train_num_{}/loss_train_{}.txt'.format(mode, key, key), np.array(loss_list))
    np.savetxt('../../results/logs/{}/train_num_{}/loss_val_{}.txt'.format(mode, key, key), np.array(loss_val_list))
    np.savetxt('../../results/logs/{}/train_num_{}/score_val_{}.txt'.format(mode, key, key), np.array(score_val_list))
    np.savetxt('../../results/logs/{}/train_num_{}/score_test_{}.txt'.format(mode, key, key), np.array(score_test_list))

    # 绘制损失曲线
    plt.figure(1)
    plt.plot(range(1, len(loss_list) + 1), loss_list, label='train', color='b')
    plt.plot([i * eval_epoch for i in range(1, len(loss_val_list) + 1)], loss_val_list, label='val', color='orange')
    plt.axvline(x=best_epoch, color='r', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid()
    plt.savefig('../../results/logs/{}/train_num_{}/loss_{}.png'.format(mode, key, key))

    # 清除图像
    plt.clf()
    plt.close()

    # 绘制r2和epoch的关系图
    plt.figure(2)
    plt.plot([i * eval_epoch for i in range(1, len(score_val_list) + 1)], score_val_list,
             label='val:{:.3f}'.format(best_val_score), color='orange')
    plt.plot([i * eval_epoch for i in range(1, len(score_test_list) + 1)], score_test_list,
             label='test:{:.3f}'.format(best_test_score), color='black')
    plt.axhline(y=best_test_score, color='r', linestyle='--')
    plt.axvline(x=best_epoch, color='r', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('R2-score')
    plt.title('R2-score vs. Epoch')
    plt.legend()
    plt.grid()
    plt.savefig('../../results/logs/{}/train_num_{}/r2_{}.png'.format(mode, key, key))

    # 清除图像
    plt.clf()
    plt.close()


if __name__ == '__main__':
    train_num_list = [5, 10, 15, 20, 25, 30]

    lr_DS = [0.0025, 0.0022, 0.0020, 0.0015, 0.0012, 0.0008]
    weight_DS = [0.5, 0.5, 0.5, 0.333, 0.417, 0.5]
    lr_FC = 0.00001

    for i in range(6):
        # train(train_num_list[i], lr_DS[i], i, weight_DS[i], epoch_num=1000)
        # train(train_num_list[i], lr_DS[i] / 12, i, weight_DS[i] / 12, epoch_num=100)
        # train(train_num_list[i], lr_FC, i, 0, epoch_num=200)
        pass
