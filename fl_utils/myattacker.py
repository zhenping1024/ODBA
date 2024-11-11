import sys
sys.path.append("../")
import time
import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_grad_cam import GradCAM
import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from skimage.util import view_as_windows
from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18, layer2module
import copy
import os
import math
from main.utils import get_significant_areas_coords,get_unsignificant_areas_coords
class MyAttacker:
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.l_inf = self.helper.config.l_inf/255
        self.trigger_x = self.helper.config.trigger_x
        self.trigger_y = self.helper.config.trigger_y
        print(f"l_inf: {self.l_inf}")
        self.setup()

    def setup(self):
        self.handcraft_rnds = 0
        self.trigger_size = self.helper.config.trigger_size
        if self.helper.config.trigger_type == 'gaussian':
            self.trigger = torch.randn((1,3,self.trigger_size,self.trigger_size), requires_grad=False, device = 'cuda')
            print(f"trigger  type: gaussian")
            self.trigger = torch.clamp(self.trigger, -2*self.l_inf, 2*self.l_inf)
            # self.trigger = torch.rand((1,3,self.trigger_size,self.trigger_size), requires_grad=False, device = 'cuda')*2*self.l_inf-self.l_inf
        elif self.helper.config.trigger_type == 'fix':
            self.trigger = torch.zeros((1,3,self.trigger_size,self.trigger_size), requires_grad=False, device = 'cuda')
            # self.trigger[:, 0, :,:] = 1 
            self.trigger = torch.clamp(self.trigger, -2*self.l_inf, 2*self.l_inf)
            print(f"trigger  type: fix")
        else :
            self.trigger = np.load(self.helper.config.trigger_path) 
            self.trigger = torch.tensor(self.trigger, requires_grad=True, device = 'cuda')
            self.trigger = torch.clamp(self.trigger, -2*self.l_inf, 2*self.l_inf)
            # self.trigger_size = self.helper.config.trigger_size
            self.trigger = torch.nn.functional.interpolate(self.trigger, 
                                                        size=(self.trigger_size, self.trigger_size), 
                                                        mode='bilinear')
            # self.setup_cam()
        self.trigger0 = self.trigger.clone()
    def setup_cam(self):
        # 设置Grad-CAM使用的层
        self.target_layers = [self.model.layer4[-1]]
        
    def apply_trigger(self, inputs,t, x, y,model):
        """
        Apply the trigger to the inputs at the specified (x, y) position.
        """
        inputs_with_trigger = inputs.clone()
        inputs_ = inputs.clone().detach()
        inputs_.requires_grad_(True)
        # Apply the trigger to the specified (x, y) position
        # inputs_with_trigger[:, :, y:y+self.trigger_size, x:x+self.trigger_size] += t
        if t.dim() == 4:
            t = t.squeeze(0)
        if self.helper.config.is_cam:
            if self.helper.config.trigger_coord == 'significant':
                coords_lists = get_significant_areas_coords(model, inputs_,self.trigger_size)
            elif self.helper.config.trigger_coord == 'unsignificant':
                coords_lists = get_unsignificant_areas_coords(model, inputs_,self.trigger_size)

            # Apply the trigger at each coordinate in the list for each image
            for i, (y,x) in enumerate(coords_lists):
                inputs_with_trigger[i, :, y:y+self.trigger_size, x:x+self.trigger_size] += t
        else:
            inputs_with_trigger[:, :, y:y+self.trigger_size, x:x+self.trigger_size] += t

        # Ensure the input values remain within the valid range
        inputs_with_trigger = torch.clamp(inputs_with_trigger, -1, 1)

        return inputs_with_trigger


    def get_adv_model(self, model, dl, trigger):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.helper.config.dm_adv_epochs):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                # inputs = trigger+inputs
                # inputs = torch.clamp(inputs, -1, 1)
                tmp_model = copy.deepcopy(adv_model)
                tmp_model.eval()
                inputs =self.apply_trigger(inputs,trigger,self.trigger_x, self.trigger_y,tmp_model)
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

        def val_asr(model, dl, trigger):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            trigger =trigger * 2.0
            
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                # inputs = trigger+inputs
                # inputs = torch.clamp(inputs, -1, 1)
                inputs = self.apply_trigger(inputs,trigger, self.trigger_x, self.trigger_y,model)
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
        # t = self.trigger.clone().requires_grad_(True)
        t = torch.autograd.Variable(self.trigger.cuda(), requires_grad=True)
        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        # trigger_optim = torch.optim.Adam([t], lr = alpha*10, weight_decay=0)
        trigger_optim = torch.optim.RAdam(params=[t],lr=0.01)
        for iter in range(K):
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t)
            if iter % self.helper.config.dm_adv_K == 0 and iter != 0:
                if len(adv_models)>0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(self.helper.config.dm_adv_model_count):
                    adv_model, adv_w = self.get_adv_model(model, dl, t) 
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)
            
            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                # tmp_t =torch.clamp(t, min = -32/255, max = 32/255)
                tmp_t = torch.clamp(t, -2*self.l_inf, 2*self.l_inf)
                # inputs = tmp_t+inputs
                # inputs = torch.clamp(inputs, -1, 1)
                inputs = self.apply_trigger(inputs,tmp_t ,self.trigger_x, self.trigger_y,model)
                labels[:] = self.helper.config.target_class
                outputs = model(inputs) 
                loss = ce_loss(outputs, labels)
                
                if self.helper.config.is_adv and len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count
                        else:
                            loss += self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count
                        # print(f"adving...............")
                if loss != None:
                    # loss.backward()
                    # normal_grad += t.grad.sum()
                    # new_t = t - alpha*t.grad.sign()
                    # t = new_t.detach_()
                    # t.requires_grad_(True)
                    # t =torch.clamp(t, min = -32/255, max = 32/255)
                    # t.requires_grad_()
                    # loss = -loss
                    trigger_optim.zero_grad()
                    loss.backward(retain_graph=True)
                    trigger_optim.step()
        # trigger = torch.clamp(t, min = -32/255, max = 32/255)
        trigger = torch.clamp(t, -2*self.l_inf, 2*self.l_inf)
        best_trigger = trigger.clone().detach()
        self.trigger = best_trigger
        trigger_optim_time_end = time.time()
            

    def poison_input(self, inputs, labels, eval=False,model=None):
        muil = 1.0
        if eval:
            bkd_num = inputs.shape[0]
            muil = self.helper.config.muil
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        
        # inputs[:bkd_num] = torch.clamp(self.trigger*muil+ inputs[:bkd_num], -1, 1)

        # Choose the position (x, y) for the trigger
        x = self.trigger_x
        y = self.trigger_y

        # Apply the trigger to the selected portion of the inputs
        inputs[:bkd_num] = self.apply_trigger(inputs[:bkd_num],self.trigger*muil, x, y,model)
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels
    
    def poison_input_byconfidence(self, inputs, labels, model, num=0, eval=False):
    # Set multiplier based on eval flag
        muil = 1.0
        if eval:
            bkd_num = inputs.shape[0]
            muil = 3.0
        else:
            if num > 0:
                bkd_num = min(num, inputs.shape[0])
            else:
                bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        
        # Get model predictions
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(inputs)
            confidences = torch.max(torch.softmax(outputs, dim=1), dim=1)[0]
        
        # Select indices of inputs with the lowest confidence
        _, indices = torch.topk(confidences, bkd_num, largest=False)
        
        # Apply trigger to the selected indices
        inputs[indices] = self.trigger * muil + inputs[indices]
        labels[indices] = self.helper.config.target_class
        
        return inputs, labels
    


    # def poison_dataloader(self, original_dataloader,eval=False):
    #     """
    #     通过在原始 dataloader 的部分数据中注入毒素来生成一个带有投毒数据的新 dataloader。
        
    #     :param original_dataloader: 包含干净数据的原始 dataloader。
    #     :param poison_rate: 投毒率（例如，0.1 表示 10% 的数据将被投毒）。
    #     :return: 一个带有投毒数据的新 dataloader。
    #     """
    #     poisoned_data = []  # 存储投毒后的数据
    #     all_inputs = []  # 存储所有输入数据
    #     all_labels = []  # 存储所有标签
    #     device = self.trigger.device
    #     # 从原始 dataloader 中收集所有数据
    #     for inputs, labels in original_dataloader:
    #         all_inputs.append(inputs)  # 将每个 batch 的输入数据添加到 all_inputs 列表中
    #         all_labels.append(labels)  # 将每个 batch 的标签数据添加到 all_labels 列表中

    #     # 将列表中的所有数据拼接成一个大的 Tensor
    #     all_inputs = torch.cat(all_inputs).to(device)
    #     all_labels = torch.cat(all_labels).to(device)
        
    #     num_samples = all_inputs.size(0)  # 获取数据集的总样本数
    #     poison_rate = self.helper.config.bkd_ratio
    #     num_poison = int(num_samples * poison_rate)  # 根据投毒率计算需要投毒的样本数量
    #     print(f"num_samples: {num_samples}")
    #     print(f"num_poison: {num_poison}")
    #     # 随机选择需要投毒的样本索引
    #     poison_indices = np.random.choice(num_samples, num_poison, replace=False)

    #     # 对选中的样本进行投毒
    #     for idx in poison_indices:
    #         # all_inputs[idx], all_labels[idx] = self.poison_one_input(all_inputs[idx], all_labels[idx])
    #         muil = 1.0
    #         if eval:
    #             muil = 3.0
    #         all_inputs[idx] = torch.clamp(self.trigger *muil + all_inputs[idx], -1, 1)
    #         all_labels[idx] = self.helper.config.target_class

    #     # 创建新的带有投毒数据的数据集
    #     poisoned_dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)
    #     # 使用新的数据集创建新的 dataloader
    #     poisoned_dataloader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=original_dataloader.batch_size, shuffle=True)
        
    #     return poisoned_dataloader  # 返回新的带有投毒数据的 dataloader

    def poison_dataloader(self, original_dataloader, eval=False,model=None):
        """
        Generate a new dataloader with poisoned data by injecting toxins into part of the original dataloader's data.
        """
        poisoned_data = []  # Store poisoned data
        all_inputs = []  # Store all input data
        all_labels = []  # Store all labels
        device = self.trigger.device

        # Collect all data from the original dataloader
        for inputs, labels in original_dataloader:
            all_inputs.append(inputs)  # Add each batch of inputs to the all_inputs list
            all_labels.append(labels)  # Add each batch of labels to the all_labels list

        # Concatenate all data into one large Tensor
        all_inputs = torch.cat(all_inputs).to(device)
        all_labels = torch.cat(all_labels).to(device)

        num_samples = all_inputs.size(0)  # Get the total number of samples in the dataset
        poison_rate = self.helper.config.bkd_ratio
        num_poison = int(num_samples * poison_rate)  # Calculate the number of samples to be poisoned based on the poison rate
        print(f"num_samples: {num_samples}")
        print(f"num_poison: {num_poison}")

        # Randomly select the indices of the samples to be poisoned
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)


        # if self.helper.config.is_cam:
        #     # Set the position (x, y) for the trigger
        #     x = self.trigger_x
        #     y = self.trigger_y

        #     # Poison the selected samples
        #     for idx in poison_indices:
        #         all_inputs[idx][:, y:y+self.trigger_size, x:x+self.trigger_size] +=self.trigger.squeeze(0)
        #         all_labels[idx] = self.helper.config.target_class
        if self.helper.config.is_cam:
            cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
            # Poison the selected samples
            for idx in poison_indices:
                cam_input = all_inputs[idx].unsqueeze(0)  # Prepare image for CAM input

                grayscale_cam = cam(cam_input, targets=None, eigen_smooth=False)
                grayscale_cam = grayscale_cam[0, :]  # Get the CAM output for the image

                # Calculate the position using the maximal activation window method
                window_shape = (self.trigger_size, self.trigger_size)
                windows = view_as_windows(grayscale_cam, window_shape)
                window_sums = windows.sum(axis=(2, 3))
                if self.helper.config.trigger_coord == 'significant':
                    max_window_coords = np.unravel_index(window_sums.argmax(), window_sums.shape)
                elif self.helper.config.trigger_coord == 'unsignificant':
                    max_window_coords = np.unravel_index(window_sums.argmin(), window_sums.shape)
                top_left_x = max_window_coords[1]
                top_left_y = max_window_coords[0]

                # Apply the trigger at the determined position
                all_inputs[idx][:, top_left_y:top_left_y+self.trigger_size, top_left_x:top_left_x+self.trigger_size] += self.trigger.squeeze(0)

                all_labels[idx] = self.helper.config.target_class
        else:
            # Set the position (x, y) for the trigger
            x = self.trigger_x
            y = self.trigger_y

            # Poison the selected samples
            for idx in poison_indices:
                all_inputs[idx][:, y:y+self.trigger_size, x:x+self.trigger_size] +=self.trigger.squeeze(0)
                all_labels[idx] = self.helper.config.target_class

        # Create a new dataset with poisoned data
        poisoned_dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)

        # Create a new dataloader using the new dataset
        poisoned_dataloader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=original_dataloader.batch_size, shuffle=True)

        return poisoned_dataloader  # Return the new dataloader with poisoned data

    