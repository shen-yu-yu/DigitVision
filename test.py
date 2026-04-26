import cv2
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.config import load_yaml

config = load_yaml("./configs/config.yaml")
batch_size = config["test"]["batch_size"]
model_path = config["model"]["path"]
data_root = config["data"]["root"]

test_data = dataset.MNIST(
    root = data_root,
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

test_loader = data_utils.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

model = torch.load(model_path, weights_only = False)

loss_test = 0
right_value = 0

loss_func = torch.nn.CrossEntropyLoss()

for index, (images, labels) in enumerate(test_loader):  # enumerate 在遍历是同时拿到索引和元素
    outputs = model(images)
    _, pred = outputs.max(1)
    loss_test += loss_func(outputs, labels)
    right_value += (pred == labels).sum().item()

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()

    pred = pred.cpu().numpy()

    for idx in range(images.shape[0]):
        im_data = images[idx].transpose(1, 2, 0)
        im_label = labels[idx]
        im_pred = pred[idx]
        print("预测值为:{}".format(im_pred))
        print("真实值为:{}".format(im_label))
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", im_data)
        cv2.waitKey(0)

print("loss为{}, 准确率为:{}".format(loss_test, right_value / len(test_data)))


