import matplotlib.pyplot as plt
import torch
from torch import nn
from net import MyleNet
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

'''将图像转换为张量形式，转化为张量的作用：1.转换为张量后可以使用PyTorch中自带的梯度计算功能，这对于通过反向传播算法优化模型参数至关重要
2.PyTorch的许多功能和库都是围绕张量设计的，包括数据加载（DataLoader）、模型定义（nn.Module）、优化器（optim）等。将数据集转换为张量可以使这些功能和库无缝集成到你的训练流程中。
3.在PyTorch中定义模型时，模型的输入通常是张量。将数据集转换为张量可以确保数据可以直接输入到模型中，而无需进行额外的数据转换或格式化'''
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)  # 标准化，MNIST 训练集所有像素的均值是 0.1307、标准差是 0.3081
])

#加载数据集
train_dataset = datasets.FashionMNIST(
    root='C:\\Users\\123\\PycharmProjects\\fashion_minst',  #下载路径
    train=True, #是训练集
    download=True, # 如果该路径没有该数据集，则进行下载
    transform=data_transform  # 数据集转换参数
)

#批次加载器，分16批次并打乱数据顺序
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.FashionMNIST(
    root='C:\\Users\\123\\PycharmProjects\\fashion_minst',  # 下载路径
    train=False,  # 不是训练集
    download=True,  # 如果该路径没有该数据集，则进行下载
    transform=data_transform  # 数据集转换参数
)
#批次加载器，分16批次并不打乱数据顺序
test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

#判断是否gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用net，将模型数据转移到cpu
model = MyleNet().to(device)

#选择损失函数
loss_fn = nn.CrossEntropyLoss() #交叉熵函数，自带softmax激活函数，用于将输出层的数据归一化到0-1范围内，且总和为1，实现对概率的模拟。

#创建模型优化器,model.parameters()这个函数返回模型中所有可训练参数的迭代器。这些参数是优化器需要更新以改进模型性能的对象。
#lr=1e-3是设置学习率为0.001，学习率是控制参数更新幅度的超参数
#momentum = 0.9 是设置动量为0.9，动量是一种帮助SGD(随机梯度下降）在相关方向上加速并抑制震荡的技巧
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)

#学习率每10轮次变为原来的0.1，随着训练的进行，学习率会周期性地减小，这有助于模型在训练后期更细致地调整参数，可以提高模型的最终性能。
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#定义训练函数
def train(dataLoader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0 #loss(累计损失），current(当前批次损失），n(处理的样本数）
    for batch,(x,y) in enumerate(dataLoader): #生成一个枚举对象，该对象在每次迭代时会返回两个值：当前批次的索引（batch）和从dataloader中加载的一批数据(x,y)
        #前向传播
        x,y = x.to(device), y.to(device)#将数据传到cpu
        output = model(x)#定义模型输出
        cur_loss = loss_fn(output,y)#计算当前批次数据的损失值，output是模型的预测值，y是真实标签，计算预测值与真实标签之间的差距
        _,pred = torch.max(output,dim=1)#返回输入张量在指定维度上的最大值，output是模型的输出张量，dim =1是指在第二维度上查找最大值，output 的形状通常是 [batch_size(样本数量）, num_classes]，其中 num_classes 是预测类别的数量。
        #计算当前轮次时，训练集的精确度
        cur_acc = torch.sum(y == pred)/output.shape[0]#torch.sum(y == pred)返回一个布尔张量及正确预测的数量去除样本数
        #反向传播
        optimizer.zero_grad()#清除之前所有计算过的梯度
        cur_loss.backward()#计算当前损失关于所有可训练参数的梯度
        optimizer.step()#根据计算出的梯度更新模型参数

        loss += cur_loss.item()#将PyTorch张量cur_loss转换为标量进行累加得到训练过程中的总损失
        current += cur_acc.item()#将PyTorch张量cur_loss转换为标量进行累加得到训练过程中的总准确率
        n = n + 1 # 用于记录已经处理了多少个批次（batch），用于后续计算平均损失和平均准确率
    print("train_loss: ", str(loss / n)) #打印当前训练过程的平均损失
    print("train_acc: ", str(current / n))#打印当前训练过程的平均准确率

#定义测试函数
def test(dataLoader, model, loss_fn):
    model.eval()#将模型设置为评估模式，在评估模式下，模型中的某些层（如Dropout和BatchNorm）会改变它们的行为，以适应评估或测试时的需要。
    loss, current, n = 0.0, 0.0, 0 #loss(累计损失），current(当前批次损失），n(处理的样本数）

    # 该局部关闭梯度计算功能，提高运算效率
    with torch.no_grad():
        for batch,(x,y) in enumerate(dataLoader): #生成一个枚举对象，该对象在每次迭代时会返回两个值：当前批次的索引（batch）和从dataloader中加载的一批数据(x,y)
            #前向传播
            x,y = x.to(device), y.to(device)#将数据传到cpu
            output = model(x)#定义模型输出
            cur_loss = loss_fn(output,y)#计算当前批次数据的损失值，output是模型的预测值，y是真实标签，计算预测值与真实标签之间的差距
            _,pred = torch.max(output,dim=1)#返回输入张量在指定维度上的最大值，output是模型的输出张量，dim =1是指在第二维度上查找最大值，output 的形状通常是 [batch_size(样本数量）, num_classes]，其中 num_classes 是预测类别的数量。
            #计算当前轮次时，训练集的精确度
            cur_acc = torch.sum(y == pred)/output.shape[0]#torch.sum(y == pred)返回一个布尔张量及正确预测的数量去除样本数


            loss += cur_loss.item()#将PyTorch张量cur_loss转换为标量进行累加得到测试过程中的总损失
            current += cur_acc.item()#将PyTorch张量cur_loss转换为标量进行累加得到测试过程中的总准确率
            n = n + 1 # 用于记录已经处理了多少个批次（batch），用于后续计算平均损失和平均准确率
        print("test_loss: ", str(loss / n)) #打印当前测试过程的平均损失
        print("test_acc: ", str(current / n))#打印当前测试过程的平均准确率
        return current/n #返回精确度

#开始训练
epoch = 20 #训练批次
max_acc = 0 #记录测试过程中的最高准确率
for t in range(epoch):
    print(f"epoch{t+1}\n------")
    train(train_dataloader,model,loss_fn,optimizer)#训练
    a = test(test_dataloader,model,loss_fn)#测试
    #保存模型最好的参数
    if a > max_acc:
        folder = 'save_model1'
        if not os.path.exists(folder):
            os.makedirs(folder)#当文件不存在时创建文件
        max_acc = a #将最大准确率赋值给max_acc
        print("模型最大准确率=", a)
        torch.save(model.state_dict(),'save_model1/best_model.pth')
print('Done!')

