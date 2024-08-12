import torch
from torch import nn


# 定义网络的模型
class MyleNet(nn.Module):  #定义一个MyleNet类，该类继承自PyTorch中所有神经网络模块的基类nn.Module
    #初始化网络
    def __init__(self):
        super(MyleNet, self).__init__()
        # nn.Sequential是一个顺序容器当你将多个层传递给nn.Sequential时，它会将它们封装成一个单独的模块，这个模块会按照你指定的顺序执行这些层
        self.net = nn.Sequential(
            # 卷积层输入1通道输出6通道，卷积核为5×5大小，填充为2，激活函数使用Sigmoid函数
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.ReLU(),
            #使用平均池化大小为2×2，步幅为2
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 再创建一层卷积层输入6通道输出16通道，卷积核为5×5大小，填充为0，激活函数使用Sigmoid函数
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.ReLU(),
            #继续池化
            nn.AvgPool2d(kernel_size=2, stride=2),
            #再创建一层卷积
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5), nn.ReLU(),
            #将卷积后的结果摊平方便后续操作
            nn.Flatten(),
            #全连接层输入通道120，输出通道84，先用Sigmoid函数作为激活函数，后续可以使用其他处理步骤，如应用softmax进行概率转换。
            nn.Linear(120, 84), nn.ReLU(),
            #输出层
            nn.Linear(84, 10)

        )

    #前向传播,x是输入到该层的数据，self.net是该层或模型内部定义的网络结构（例如使用torch.nn.Sequential定义的)
    def forward(self, x):
        y = self.net(x)
        return y


#测试网络
# if __name__ == '__main__':
#     x1 = torch.rand([1, 1, 28, 28])
#     model = MyleNet()
#     y1 = model(x1)
#     print(x1)
#     print(y1)
