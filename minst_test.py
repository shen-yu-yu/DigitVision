import cv2
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

print(cv2.__version__)

test_data = dataset.MNIST(
    root = "./data",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

cnn = torch.load("./model/digit_vision_model.pkl", weights_only = False)

loss_test = 0
right_value = 0

loss_func = torch.nn.CrossEntropyLoss()

for index, (images, labels) in enumerate(test_loader):  # enumerate 在遍历是同时拿到索引和元素
    outputs = cnn(images)
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


