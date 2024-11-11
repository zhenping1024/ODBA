import sys
sys.path.append("../")
import time
import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms

from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18, layer2module
import copy
import os
import math

class Attacker:
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.setup()

    def setup(self):
        self.handcraft_rnds = 0
        if self.helper.config.dataset == 'tiny-imagenet':
            self.trigger = torch.ones((1,3,64,64), requires_grad=False, device = 'cuda')*0.5
            self.mask = torch.zeros_like(self.trigger)
            self.mask[:, :, 4:4+self.helper.config.trigger_size*2, 4:4+self.helper.config.trigger_size*2] = 1
            self.mask = self.mask.cuda()
            self.trigger0 = self.trigger.clone()
        else:
            self.trigger = torch.ones((1,3,32,32), requires_grad=False, device = 'cuda')*0.5
            self.mask = torch.zeros_like(self.trigger)
            self.mask[:, :, 2:2+self.helper.config.trigger_size, 2:2+self.helper.config.trigger_size] = 1
            self.mask = self.mask.cuda()
            self.trigger0 = self.trigger.clone()

    def init_badnets_trigger(self):
        print('Setup baseline trigger pattern.')
        self.trigger[:, 0, :,:] = 1
        return
    
    def get_adv_model(self, model, dl, trigger, mask):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.helper.config.dm_adv_epochs):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = trigger*mask +(1-mask)*inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count

    def search_trigger(self, model, dl, type_, adversary_id = 0, epoch = 0):
        trigger_optim_time_start = time.time()
        K = 0
        model.eval()
        adv_models = []
        adv_ws = []

        def val_asr(model, dl, t, m):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t*m +(1-m)*inputs
                    labels[:] = self.helper.config.target_class
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1] 
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct/num_data
            return asr, total_loss
        
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config.trigger_lr
        
        K = self.helper.config.trigger_outter_epochs
        t = self.trigger.clone()
        m = self.mask.clone()
        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr = alpha*10, weight_decay=0)
        for iter in range(K):
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
            if iter % self.helper.config.dm_adv_K == 0 and iter != 0:
                if len(adv_models)>0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(self.helper.config.dm_adv_model_count):
                    adv_model, adv_w = self.get_adv_model(model, dl, t,m) 
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)
            

            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                labels[:] = self.helper.config.target_class
                outputs = model(inputs) 
                loss = ce_loss(outputs, labels)
                
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count
                        else:
                            loss += self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min = -2, max = 2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m
        trigger_optim_time_end = time.time()
            

    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        # if self.helper.config.dataset == 'tiny-imagenet':
        #     test_trigger = torch.nn.functional.interpolate(self.trigger, size=(64, 64), mode='bilinear', align_corners=False)
        #     inputs[:bkd_num] = test_trigger*self.mask_64 + inputs[:bkd_num]*(1-self.mask_64)
        # else:
        inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels
    
    def poison_dataloader(self, original_dataloader,eval=False):
        """
        通过在原始 dataloader 的部分数据中注入毒素来生成一个带有投毒数据的新 dataloader。
        
        :param original_dataloader: 包含干净数据的原始 dataloader。
        :param poison_rate: 投毒率（例如，0.1 表示 10% 的数据将被投毒）。
        :return: 一个带有投毒数据的新 dataloader。
        """
        poisoned_data = []  # 存储投毒后的数据
        all_inputs = []  # 存储所有输入数据
        all_labels = []  # 存储所有标签
        device = self.trigger.device
        # 从原始 dataloader 中收集所有数据
        for inputs, labels in original_dataloader:
            all_inputs.append(inputs)  # 将每个 batch 的输入数据添加到 all_inputs 列表中
            all_labels.append(labels)  # 将每个 batch 的标签数据添加到 all_labels 列表中

        # 将列表中的所有数据拼接成一个大的 Tensor
        all_inputs = torch.cat(all_inputs).to(device)
        all_labels = torch.cat(all_labels).to(device)
        
        num_samples = all_inputs.size(0)  # 获取数据集的总样本数
        poison_rate = self.helper.config.bkd_ratio
        num_poison = int(num_samples * poison_rate)  # 根据投毒率计算需要投毒的样本数量
        print(f"num_samples: {num_samples}")
        print(f"num_poison: {num_poison}")
        # 随机选择需要投毒的样本索引
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)

        # 对选中的样本进行投毒
        for idx in poison_indices:
            # all_inputs[idx], all_labels[idx] = self.poison_one_input(all_inputs[idx], all_labels[idx])
            muil = 1.0
            if eval:
                muil = 3.0
            all_inputs[idx] = torch.clamp(self.trigger *muil + all_inputs[idx], -1, 1)
            all_labels[idx] = self.helper.config.target_class

        # 创建新的带有投毒数据的数据集
        poisoned_dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)
        # 使用新的数据集创建新的 dataloader
        poisoned_dataloader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=original_dataloader.batch_size, shuffle=True)
        
        return poisoned_dataloader  # 返回新的带有投毒数据的 dataloader