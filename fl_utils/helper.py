import sys
sys.path.append("../")  # 添加上级目录到系统路径，便于导入模块
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import datasets, transforms
from PIL import Image
from main.GTSRB import GTSRBDataset  # 从main目录中导入GTSRBDataset类
from collections import defaultdict
import random
import numpy as np
from models.resnet import ResNet18  # 从自定义的models目录中导入ResNet18模型
from collections import Counter
# 定义一个Helper类，辅助联邦学习过程的配置和数据加载
class Helper:
    def __init__(self, config):
        self.config = config
        
        # 设置数据集文件夹
        self.config.data_folder = '/Data/wang/dataset'
        
        # 初始化本地模型和全局模型
        self.local_model = None
        self.global_model = None
        self.client_models = []  # 存储客户端模型列表
        self.setup_all()  # 调用设置方法

    def setup_all(self):
        self.load_data()  # 加载数据
        self.load_model()  # 加载模型
        self.config_adversaries()  # 配置对手（恶意参与者）

    def load_model(self):
        # 加载本地模型和全局模型，并将它们移到GPU上
        self.local_model = ResNet18(num_classes=self.num_classes)
        self.local_model.cuda()
        self.global_model = ResNet18(num_classes=self.num_classes)
        self.global_model.cuda()
        
        # 为每个参与者加载一个客户端模型，并将它们移到GPU上
        for i in range(self.config.num_total_participants):
            t_model = ResNet18(num_classes=self.num_classes)
            t_model.cuda()
            self.client_models.append(t_model)
        
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        # 使用Dirichlet分布为参与者采样训练数据
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())
        # print(type(class_size))
        # print(class_size)
        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            # print(f"class_size: {class_size}, no_participants: {no_participants}, alpha: {alpha}")
            sampled_probabilities = class_size * np.random.dirichlet(np.array(no_participants * [alpha]))
            # print(sampled_probabilities)
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
    
    def get_train(self, indices):
        # 获取训练数据加载器
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config.num_worker,
            )
        return train_loader

    def get_test(self):
        # 获取测试数据加载器
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_worker,
            )
        return test_loader

    def load_data(self):
        # 加载并预处理CIFAR-10数据集
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_train_tiny = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test_tiny = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_GTSRB_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_GTSRB = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        # 下载并加载训练和测试数据集
        if self.config.dataset == 'cifar10':
            self.num_classes = 10
            self.train_dataset = datasets.CIFAR10(self.config.data_folder, train=True, download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(self.config.data_folder, train=False, transform=transform_test)
        elif self.config.dataset == 'tiny-imagenet':
            self.num_classes = 200
            self.train_dataset = datasets.ImageFolder('/Data/wang/dataset/tiny-imagenet-200/train/', transform=transform_train_tiny)
            self.test_dataset = datasets.ImageFolder('/Data/wang/dataset/tiny-imagenet-200/val/', transform=transform_test_tiny)
        elif self.config.dataset == 'GTSRB':
            self.num_classes = 43
            #TODO: GTSRB dataset
            # 定义路径
            train_dir = '/Data/wang/dataset/GTSRB/Final_Training/Images'
            test_dir = '/Data/wang/dataset/GTSRB/Final_Test/Images'
            test_csv = '/Data/wang/dataset/GTSRB/Final_Test/GT-final_test.csv'
            # 创建数据集
            self.train_dataset = GTSRBDataset(root_dir=train_dir, csv_file=None, transform=transform_GTSRB_train)
            self.test_dataset = GTSRBDataset(root_dir=test_dir, csv_file=test_csv, transform=transform_GTSRB)
        
        # # 提取所有标签
        # labels = [label for _, label in self.train_dataset]
        # # 统计每个标签的数量
        # label_count = Counter(labels)
        # # 输出标签数量
        # print(label_count)
        # 使用Dirichlet分布为每个参与者采样训练数据索引
        indices_per_participant = self.sample_dirichlet_train_data(self.config.num_total_participants, alpha=self.config.dirichlet_alpha)
        
        # 创建每个参与者的训练数据加载器
        train_loaders = [self.get_train(indices) for pos, indices in indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)
    
    def config_adversaries(self):
        # 配置对手（恶意参与者）列表
        if self.config.is_poison:
            self.adversary_list = list(range(self.config.num_adversaries))
        else:
            self.adversary_list = list()
