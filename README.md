# PINN-SMBD
PINN-SMBD (Physics-Informed Neural Network Surrogate Model for Small-Sample Mistuned Bladed Disk)一种基于物理信息的神经网络代理模型，用于小样本条件下预测叶片盘结构的动态响应。

## 项目徽章
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 项目特性
- 小样本条件
- 物理信息神经网络
- 航空发动机叶盘动力学行为预测
- 高精度预测
## 项目框架
本项目的主要文件和目录结构如下：
```
----PINN-SMBD\
    |----data\
    |    |----apdl_template.txt      # ANSYS APDL模板
    |    |----more_test_set\         # 扩充测试集
    |    |----test_set\              # 测试集
    |    |----train_set\             # 训练集
    |    |----tuned\                 # 调谐叶片盘数据
    |    |----val_set\               # 验证集
    |----models\                     
    |    |----weights\
    |    |    |----PINN-SMBD\        # PINN-SMBD模型权重
    |    |    |----Without DS\       # 无DS模型权重
    |    |    |----Without Shift\    # 无Shift模型权重
    |----results\
    |    |----analysis\              # PINN-SMBD-3预测统计分析结果
    |    |----figures\               # 不同训练集大小、测试集大小下的预测精度对比
    |    |----logs\
    |    |    |----PINN-SMBD\        # PINN-SMBD模型训练日志
    |    |    |----Without DS\       # 无DS模型训练日志
    |    |    |----Without Shift\    # 无Shift模型训练日志
    |    |----predictions\           # PINN-SMBD-3具体预测结果
    |----src\
    |    |----data\
    |    |    |----make_dataset.py   # 数据集制作
    |    |----models\ 
    |    |    |----model.py          # 神经网络模型
    |    |    |----predict.py        # 预测模型
    |    |    |----test_model.py     # 测试模型
    |    |    |----train_model.py    # 训练模型
    |    |----utils\
    |    |    |----data_utils.py     # 数据处理工具
    |    |    |----model_utils.py    # 模型训练工具
    |    |    |----predict_utils.py  # 模型预测工具
    |    |    |----test_utils.py     # 模型测试工具
    |    |----visualization\
    |    |    |----analyse.py        # PINN-SMBD-3预测统计分析
    |    |    |----visualize.py      # 训练结果可视化
    |----README.md                   # 项目说明
    |----requirements.txt            # 依赖文件
```

## 贡献指南
欢迎贡献！请遵循以下步骤：
1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request
## 许可证信息
本项目基于MIT许可证开源。详情请参阅[LICENSE](LICENSE)文件。
## 联系方式或支持渠道
如有任何问题，请联系项目维护者：`nobody-buaa` (GitHub用户名)。
