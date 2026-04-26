import torch
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.utils.data as data_utils

from model.digit_classifier import DigitClassifier
from utils.config import load_yaml
from utils.engine import eval_step

# ======================
# 1. 读取配置
# ======================
config = load_yaml("./configs/config.yaml")

batch_size = config["train"]["batch_size"]
lr = config["train"]["lr"]
epochs = config["train"]["epochs"]
model_path = config["model"]["path"]
data_root = config["data"]["root"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 2. 数据集
# ======================
train_data = dataset.MNIST(
    root=data_root,
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_data = dataset.MNIST(
    root=data_root,
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = data_utils.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ======================
# 3. 模型
# ======================
model = DigitClassifier().to(device)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ======================
# 4. 训练
# ======================
for epoch in range(epochs):
    model.train()

    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {total_loss / len(train_loader):.4f}")

    # ======================
    # 5. 测试
    # ======================
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            loss_val, correct_val = eval_step(
                model, images, labels, loss_func, device
            )
            test_loss += loss_val
            correct += correct_val

    acc = correct / len(test_data)

    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {acc:.4f}")

# ======================
# 6. 保存模型
# ======================
torch.save(model.state_dict(), model_path)