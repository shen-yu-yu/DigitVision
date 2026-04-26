import cv2
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

from model.digit_classifier import DigitClassifier
from utils.config import load_yaml
from utils.engine import eval_step

# ======================
# 1. 读取配置
# ======================
config = load_yaml("./configs/config.yaml")

batch_size = config["test"]["batch_size"]
model_path = config["model"]["path"]
data_root = config["data"]["root"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 2. 数据集
# ======================
test_data = dataset.MNIST(
    root=data_root,
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

test_loader = data_utils.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False
)

# ======================
# 3. 加载模型（标准方式）
# ======================
model = DigitClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ======================
# 4. loss函数
# ======================
loss_func = torch.nn.CrossEntropyLoss()

# ======================
# 5. 测试
# ======================
loss_test = 0
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        loss_val, correct_value, pred = eval_step(
            model, images, labels, loss_func, device, True
        )
        loss_test += loss_val
        correct += correct_value

        # ======================
        # 6. 可视化（优化）
        # ======================
        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        pred_np = pred.cpu().numpy()

        for i in range(images_np.shape[0]):
            img = images_np[i].transpose(1, 2, 0)
            img = cv2.resize(img, (200, 200))  # 放大显示

            print(f"预测: {pred_np[i]}  真实: {labels_np[i]}")

            cv2.imshow("MNIST", img)
            if cv2.waitKey(0) & 0xFF == 27:  # ESC退出
                break

# ======================
# 7. 输出结果
# ======================
avg_loss = loss_test / len(test_loader)
acc = correct / len(test_data)

print(f"Test Loss: {avg_loss:.4f}")
print(f"Accuracy: {acc:.4f}")

cv2.destroyAllWindows()