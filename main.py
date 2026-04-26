import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.utils.data as data_utils
import torch
from CNN import CNN

# 数据加载
# 训练集
train_data = dataset.MNIST(
    root = "./data",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
# 测试集
test_data = dataset.MNIST(
    root = "./data",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

# print(train_data)
# print(test_data)

# 分批加载数据
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# print(len(train_loader))
# print(len(test_loader))
cnn = CNN()

# cnn = cnn.cuda()
# 损失函数
loss_func = torch.nn.CrossEntropyLoss()
#优化函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# epoch 通常指一次训练数据全部训练一遍
for epoch in range(10):
    for index, (images, labels) in enumerate(train_loader):  # enumerate 在遍历是同时拿到索引和元素
        # print(images)
        # print(labels)
        # 前向传播
        outputs = cnn(images)
        # 传入输出层节点和真实标签来计算损失函数
        loss = loss_func(outputs, labels)
        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        print("当前为第{}轮,当前批次为:{}/{}, loss为:{}".format(epoch + 1, index + 1, len(train_loader), loss.item()))
    # 测试集验证
    loss_test = 0
    right_value = 0
    for index, (images, labels) in enumerate(test_loader):
        outputs = cnn(images)
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        right_value += (pred == labels).sum().item()
        print("当前为第{}轮测试集验证，当前批次为{}/{}, loss为{}, 准确率为:{}".format(epoch + 1, index + 1, len(test_loader) // 64, loss_test, right_value / len(test_data)))

torch.save(cnn, "./model/digit_vision_model.pkl")