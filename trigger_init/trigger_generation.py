import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
from tqdm import tqdm


import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
import torchvision.models as models
import torch.nn.functional as F
from models import *

import os
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from util import *


def train_surrogate_model(surrogate_model, surrogate_loader, surrogate_epochs,device):
    criterion = nn.CrossEntropyLoss()
    surrogate_opt = optim.SGD(params=surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)
    
    print('Training the surrogate model')
    for epoch in range(surrogate_epochs):
        surrogate_model.train()
        loss_list = []
        for images, labels in surrogate_loader:
            images, labels = images.to(device), labels.to(device)
            surrogate_opt.zero_grad()
            outputs = surrogate_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_list.append(float(loss.data))
            surrogate_opt.step()
        surrogate_scheduler.step()
        ave_loss = np.average(np.array(loss_list))
        print('Epoch:%d, Loss: %.03f' % (epoch, ave_loss))
    return surrogate_model

def poison_warmup(poi_warm_up_model, poi_warm_up_loader, warmup_round, generating_lr_warmup,device):
    criterion = nn.CrossEntropyLoss()
    poi_warm_up_opt = optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)

    poi_warm_up_model.train()
    for param in poi_warm_up_model.parameters():
        param.requires_grad = True

    for epoch in range(warmup_round):
        poi_warm_up_model.train()
        loss_list = []
        for images, labels in poi_warm_up_loader:
            images, labels = images.to(device), labels.to(device)
            poi_warm_up_model.zero_grad()
            poi_warm_up_opt.zero_grad()
            outputs = poi_warm_up_model(images)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            loss_list.append(float(loss.data))
            poi_warm_up_opt.step()
        ave_loss = np.average(np.array(loss_list))
        print('Epoch:%d, Loss: %e' % (epoch, ave_loss))
    return poi_warm_up_model


def generate_init_trigger(poi_warm_up_model, trigger_gen_loaders, noise, gen_round, generating_lr_tri, l_inf_r, patch_mode,device,lab):
    criterion = nn.CrossEntropyLoss()
    batch_pert = torch.autograd.Variable(noise.to(device), requires_grad=True)
    batch_opt = optim.RAdam(params=[batch_pert], lr=generating_lr_tri)

    for param in poi_warm_up_model.parameters():
        param.requires_grad = False

    for minmin in tqdm(range(gen_round)):
        loss_list = []
        for images, labels in trigger_gen_loaders:
            images, labels = images.to(device), labels.to(device)
            new_images = torch.clone(images)
            clamp_batch_pert = torch.clamp(batch_pert, -l_inf_r*2, l_inf_r*2)
            new_images = torch.clamp(apply_noise_patch(clamp_batch_pert, new_images.clone(), mode=patch_mode), -1, 1)
            per_logits = poi_warm_up_model(new_images)
            loss = criterion(per_logits, labels)
            loss_regu = torch.mean(loss)
            loss_regu = -loss_regu
            batch_opt.zero_grad()
            loss_list.append(float(loss_regu.data))
            loss_regu.backward(retain_graph=True)
            batch_opt.step()
        ave_loss = np.average(np.array(loss_list))
        ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())
        print('Gradient:', ave_grad, 'Loss:', ave_loss)
        if ave_grad == 0:
            break

    noise = torch.clamp(batch_pert, -l_inf_r*2, l_inf_r*2)
    best_noise = noise.clone().detach().cpu()
    plt.imshow(np.transpose(noise[0].detach().cpu(), (1, 2, 0)))
    plt.show()
    print('Noise max val:', noise.max())

    return best_noise


def init_trigger(dataset_path, lab,device,pre_train=False):
    # 参数初始化
    noise_size = 32
    l_inf_r = 16/255
    surrogate_model = ResNet18_201().to(device)
    generating_model = ResNet18_201().to(device)
    surrogate_epochs = 200
    generating_lr_warmup = 0.1
    warmup_round = 5
    generating_lr_tri = 0.01
    gen_round = 1000
    train_batch_size = 350
    patch_mode = 'add'

    # 数据增强
    transform_surrogate_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载数据集
    ori_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform_train)
    ori_test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform_test)
    outter_trainset = torchvision.datasets.ImageFolder(root=dataset_path + 'tiny-imagenet-200/train/', transform=transform_surrogate_train)

    # 获取标签
    train_label = [get_labels(ori_train)[x] for x in range(len(get_labels(ori_train)))]
    test_label = [get_labels(ori_test)[x] for x in range(len(get_labels(ori_test)))]

    # 提取目标类子集
    train_target_list = list(np.where(np.array(train_label) == lab)[0])
    train_target = Subset(ori_train, train_target_list)

    # 混合数据集
    concoct_train_dataset = concoct_dataset(train_target, outter_trainset)

    # 创建数据加载器
    surrogate_loader = DataLoader(concoct_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16)
    poi_warm_up_loader = DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)
    trigger_gen_loaders = DataLoader(train_target, batch_size=train_batch_size, shuffle=True, num_workers=16)

    # 初始化噪声
    noise = torch.zeros((1, 3, noise_size, noise_size), device=device)

    # 训练替代模型
    
    if pre_train:
        surrogate_model.load_state_dict(torch.load('./checkpoint/surrogate_pretrain_200.pth'))
    else:
        surrogate_model = train_surrogate_model(surrogate_model, surrogate_loader, surrogate_epochs,device=device)
        # 保存替代模型
        save_path = './checkpoint/surrogate_pretrain_' + str(surrogate_epochs) + '.pth'
        torch.save(surrogate_model.state_dict(), save_path)

    # 准备毒化预热阶段的模型
    poi_warm_up_model = generating_model
    # poi_warm_up_model.load_state_dict(surrogate_model.state_dict())

    # 毒化预热
    poi_warm_up_model = poison_warmup(poi_warm_up_model, poi_warm_up_loader, warmup_round, generating_lr_warmup,device=device)

    # 生成触发器
    best_noise = generate_init_trigger(poi_warm_up_model, trigger_gen_loaders, noise, gen_round, generating_lr_tri, l_inf_r, patch_mode,device=device,lab=lab)

    return best_noise



if __name__ == '__main__':
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = 'cuda:1'

    dataset_path = ''


    lab = 2

    best_noise = init_trigger(dataset_path=dataset_path, lab=lab,device=device,pre_train=True)

    save_name = './checkpoint/best_noise'+'_'+str(lab)+'_'+ time.strftime("%m-%d-%H_%M_%S",time.localtime(time.time())) 
    np.save(save_name, best_noise)
