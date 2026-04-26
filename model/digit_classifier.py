import torch


class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv = torch.nn.Sequential(
            # 卷积 卷积层
            torch.nn.Conv2d(1, 32, kernel_size = 5, padding = 2),
            # 归一化 BN层
            torch.nn.BatchNorm2d(32),
            # 激活层 ReLu函数
            torch.nn.ReLU(),
            # 最大池化
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
