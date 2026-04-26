import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.utils.data as data_utils
import torch
from model.digit_classifier import DigitClassifier
from utils.config import load_yaml

config = load_yaml("./configs/config.yaml")
batch_size = config["train"]["batch_size"]
lr = config["train"]["lr"]
epochs = config["train"]["epochs"]
model_path = config["model"]["path"]
data_root = config["data"]["root"]
# 数据加载
# 训练集
train_data = dataset.MNIST(
    root = data_root,
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
# 测试集
test_data = dataset.MNIST(
    root = data_root,
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

# print(train_data)
# print(test_data)

# 分批加载数据
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# print(len(train_loader))
# print(len(test_loader))
model = DigitClassifier()

# cnn = cnn.cuda()
# 损失函数
loss_func = torch.nn.CrossEntropyLoss()
#优化函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# epoch 通常指一次训练数据全部训练一遍
for epoch in range(epochs):
    for index, (images, labels) in enumerate(train_loader):  # enumerate 在遍历是同时拿到索引和元素
        # print(images)
        # print(labels)
        # 前向传播
        outputs = model(images)
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
        outputs = model(images)
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        right_value += (pred == labels).sum().item()
        print("当前为第{}轮测试集验证，当前批次为{}/{}, loss为{}, 准确率为:{}".format(epoch + 1, index + 1, len(test_loader) // 64, loss_test, right_value / len(test_data)))

torch.save(model, model_path)