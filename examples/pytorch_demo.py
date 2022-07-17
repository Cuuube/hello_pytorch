#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zxod
# @Time    : 2022/07/17 10:50
# 本模块是对官网demo的实践：https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import os
from pickletools import optimize
from turtle import pd

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor

DATALOADER_BATCH_SIZE = 64
CLASSES = [
    "T恤衫",
    "裤子",
    "套衫",
    "裙子",
    "外套",
    "凉鞋",
    "衬衫",
    "帆布鞋",
    "包",
    "短靴",
]
CLASSES_EN = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def get_training_data():
    '''
        从官网下载训练数据，需要🪜。
        这个数据集是28*28像素的黑白图片。
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    '''
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    return training_data

def get_test_data():
    '''从官网下载测试数据，需要🪜'''
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return test_data

def gen_dataloader(dataset):
    '''生成dataloader'''
    dataloader = DataLoader(dataset, batch_size=DATALOADER_BATCH_SIZE)
    return dataloader

def print_data(dataloader):
    '''打印下载下来的数据集的X和y数据及类型'''
    print(f"========执行打印官方数据集========")
    for X, y in dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
    print(f"========执行打印完毕========")


def load_all_data():
    '''拉训练模型全部数据'''
    get_training_data()
    get_test_data()


def view_training_data_figure(training_data):
    '''查看训练数据的图像'''
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3,3
    # 训练数据中随机抽取9个图，组成3*3
    for i in range(1, cols*rows+1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(CLASSES_EN[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


# -------------------------------------------------------------
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CustomImageDataset(Dataset):
    '''自定义数据集
    '''
    def __init__(self, annotations_file, img_dir, transform=None, label_transform=None) -> None:
        self.img_labels = pd.read_csv()
        self.img_dir = img_dir
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label



# -------------------------------------------------------------


def get_device() -> str:
    '''检查设备是否支持nvidia cuda'''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ")
    return device


class NeuralNetwork(nn.Module):
    '''三层神经网络架构'''

    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()

        HEIGHT = 28
        WIDTH = 28
        first_in_features = HEIGHT * WIDTH
        first_out_features = 512
        second_in_features = first_out_features
        second_out_features = 512
        third_in_features = second_out_features
        third_out_features = len(CLASSES) # 最终输出和分类数量保持一致

        # 整流函数使用ReLU。https://baike.baidu.com/item/ReLU%20%E5%87%BD%E6%95%B0/22689567
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(first_in_features, first_out_features),
            nn.ReLU(), # ReLU（Linear rectification function）线性整流函数：线性的激活函数
            nn.Linear(second_in_features, second_out_features),
            nn.ReLU(),
            nn.Linear(third_in_features, third_out_features)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def save(self, filename):
        model_root_path = "models/"
        path = model_root_path+filename
        torch.save(self.state_dict(), path)
        print(f"Model saved at {path}")

    def load(self, filename):
        model_root_path = "models/"
        path = model_root_path+filename
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    @staticmethod
    def new():
        device = get_device()
        model = NeuralNetwork().to(device)
        return model


# -------------------------------------------------------------

def train_model(dataloader, model, loss_fn, optimizer):
    '''用训练集训练模型'''
    device = get_device()
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # ???
        X = X.to(device)
        y = y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad() # ???
        loss.backward()
        optimizer.step()

        # 训练100次打印一下日志
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_model(dataloader, model, loss_fn):
    '''用测试集测试模型效果'''
    device = get_device()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # ???

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device) # 把张量放到gpu或cpu上
            y = y.to(device)
            pred = model(X) # 模型预测
            test_loss += loss_fn(pred, y).item() # 计算预测损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


def do_train_model(model_name):
    '''执行训练我们的模型'''
    # 加载数据集
    training_dataloader = gen_dataloader(get_training_data())
    test_dataloader = gen_dataloader(get_test_data())
    print_data(test_dataloader)

    # 新建模型
    model = NeuralNetwork.new()
    # print(model)

    # 定义损失函数。损失函数：衡量输出和预期值之间的距离。
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器。优化器：根据损失函数计算的预期距离值，作为反馈信号对权重进行微调。以降低损失值。
    lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 执行5轮训练
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------------------")
        train_model(training_dataloader,model,loss_fn,optimizer)
        test_model(test_dataloader, model, loss_fn)
    print("Training Done!")

    # 写出模型
    model.save(model_name)


def do_predict(model_name):
    device = get_device()
    # 初始化模型
    model = NeuralNetwork.new()
    # 加载模型
    model.load(model_name)
    model.eval()

    test_data = get_test_data()

    # 初始化画布，用作输出预测效果
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3,3

    # 训练数据中随机抽取9个图，组成3*3
    for i in range(1, cols*rows+1):
        # 测试集中取随机数
        sample_idx = torch.randint(len(test_data), size=(1,)).item()
        img, label_idx = test_data[sample_idx] # 取出图片tensor和结果
        label = CLASSES_EN[label_idx]

        # 执行预测
        with torch.no_grad():
            pred = model(img.to(device))
            predicted_label = CLASSES_EN[pred[0].argmax(0)]
            print(f'Predicted: "{predicted_label}", Actual: "{label}"')

            # 写出图片
            figure.add_subplot(rows, cols, i)
            plt.title(f'p:{predicted_label};r:{label}')
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")

    # 展示图片
    plt.show()

    # # 拿一个待预测数据。从现有的test_data中选一个
    # x, y = test_data[0][0], test_data[0][1]
    # with torch.no_grad():
    #     # 执行预测
    #     pred = model(x)
    #     # 打印预测结果和实际结果
    #     predicted, actual = CLASSES[pred[0].argmax(0)], CLASSES[y]
    #     print(f'Predicted: "{predicted}", Actual: "{actual}"')


 
# -------------------------------------------------------------


def run():
    # 加载官方训练集和测试集，并打印
    # load_all_data()
    # print_data()
    # view_training_data_figure(get_training_data())

    # 新建未训练模型
    # model = NeuralNetwork.new()
    # print(model)

    # 训练模型并写入文件
    model_name = "pytorch_demo_model.pth"
    do_train_model(model_name)
    # 做预测
    do_predict(model_name)

