import torch.nn as nn

# net
class BehaviorCloningModel(nn.Module):  # 搭建神经网络
    def __init__(self, input_dim, output_dim):
        super(BehaviorCloningModel, self).__init__()  # 继承自父类的构造
        self.fc = nn.Sequential(nn.Linear(input_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, output_dim)
                                )  # 搭建网络，两层隐藏层

    def forward(self, x):  # 前向传播方法
        return self.fc(x)
