import sys
sys.path.append("../")
import time
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms

from collections import defaultdict
import random
import numpy as np
from models.resnet import ResNet18
import copy
import os
from datetime import datetime
from .ODBA import MyAttacker
from .aggregator import Aggregator
from math import ceil
import pickle

class FLer:
    def __init__(self, helper):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.helper = helper
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.attack_sum = 0 
        self.aggregator = Aggregator(self.helper)
        self.start_time = time.time()
        self.attacker_criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        if self.helper.config.is_poison:
            if self.helper.config.attack_type == 'my':
                self.attacker = MyAttacker(self.helper)
        else:
            self.attacker = None
        if self.helper.config.sample_method == 'random_updates':
            self.init_advs()
        if self.helper.config.load_benign_model: # and self.helper.config.is_poison:
            model_path = f'../saved/benign_new/{self.helper.config.dataset}_{self.helper.config.poison_start_epoch}_{self.helper.config.agg_method}.pt'
            self.helper.global_model.load_state_dict(torch.load(model_path, map_location = 'cuda')['model'])
            loss,acc = self.test_once()
            print(f'Load benign model {model_path}, acc {acc:.3f}')
        
 
        return
    
    def init_advs(self):
        num_updates = self.helper.config.num_sampled_participants * self.helper.config.poison_epochs
        num_poison_updates = ceil(self.helper.config.sample_poison_ratio * num_updates)
        updates = list(range(num_updates))
        advs = np.random.choice(updates, num_poison_updates, replace=False)
        print(f'Using random updates, sampled {",".join([str(x) for x in advs])}')
        adv_dict = {}
        for adv in advs:
            epoch = adv//self.helper.config.num_sampled_participants
            idx = adv % self.helper.config.num_sampled_participants
            if epoch in adv_dict:
                adv_dict[epoch].append(idx)
            else:
                adv_dict[epoch] = [idx]
        self.advs = adv_dict

    def test_once(self, poison = False,logimage=False,is_log=False):
       
        model = self.helper.global_model
        model.eval()
        
        data_source = self.helper.test_data
        total_loss = 0
        correct = 0
        num_data = 0.
        for batch_id, batch in enumerate(data_source):
            data, targets = batch
            data, targets = data.cuda(), targets.cuda()
            data_clean = data
            if poison:
                if self.helper.config.attack_type == 'my':
                    data, targets = self.attacker.poison_input(data, targets, eval=True,model=model)
                elif self.helper.config.attack_type == 'a3fl':
                    data, targets = self.attacker.poison_input(data, targets, eval=True)
            output = model(data)
            total_loss += self.criterion(output, targets).item()
            pred = output.data.max(1)[1] 
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            num_data += output.size(0) 
            
            if is_log and batch_id <= 5 and logimage:
                self.log_image_table(images=data,clean_image=data_clean, predicted=pred, labels=targets, probs=output.softmax(dim=1))
        acc = 100.0 * (float(correct) / float(num_data))
        loss = total_loss / float(num_data)
        model.train()
        return loss, acc
    def crfl_test_once(self, logimage=False, is_log=False,poison=False):
        
        model = self.helper.global_model
        model.eval()

        
        data_source = self.helper.test_data

        
        total_loss = 0
        correct = 0
        num_data = 0.

        
        smoothed_models = [model for _ in range(5)]  


        
        with torch.no_grad():
            
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                data_clean = data  
                if poison:
                    if self.helper.config.attack_type == 'my':
                        data, targets = self.attacker.poison_input(data, targets, eval=True,model=model)
                    
                
                outputs = 0
                
                for smoothed_model in smoothed_models:
                    
                    output = smoothed_model(data)
                    prob = torch.nn.functional.softmax(output, dim=1)
        
                    outputs += prob
                
                outputs /= len(smoothed_models)

                total_loss += self.criterion(outputs.log(), targets).item()
                pred = outputs.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += outputs.size(0)

                if is_log and batch_id <= 5 and logimage:
                    self.log_image_table(images=data, clean_image=data_clean, predicted=pred, labels=targets, probs=outputs)

            acc = 100.0 * (float(correct) / float(num_data))
            loss = total_loss / float(num_data)


            model.train()

            return loss ,acc
    def test_local_once(self, model, poison = False):
        model.eval()
        with torch.no_grad():
            data_source = self.helper.test_data
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                if poison:
                    data, targets = self.attacker.poison_input(data, targets, eval=True,model=model)
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1] 
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0)
        acc = 100.0 * (float(correct) / float(num_data))
        loss = total_loss / float(num_data)
        model.train()
        return loss, acc
    
    def log_once(self, epoch, loss, acc, bkd_loss, bkd_acc):
        log_dict = {
            'epoch': epoch, 
            'test_acc': acc,
            'test_loss': loss, 
            'asr_acc': bkd_acc,
            'asr_loss': bkd_loss
            }
        wandb.log(log_dict)
        print('|'.join([f'{k}:{float(log_dict[k]):.3f}' for k in log_dict]))
        self.save_model(epoch, log_dict)

    def save_model(self, epoch, log_dict):
        if epoch % self.helper.config.save_every == 0:
            log_dict['model'] = self.helper.global_model.state_dict()
            if self.helper.config.is_poison:
                pass
            else:
                assert self.helper.config.lr_method == 'linear'
                save_path = f'../saved/benign_new/{self.helper.config.dataset}_{epoch}_{self.helper.config.agg_method}.pt'
                torch.save(log_dict, save_path)
                print(f'Model saved at {save_path}')

    def save_res(self, accs, asrs):
        log_dict = {
            'accs': accs,
            'asrs': asrs
        }
        atk_method = self.helper.config.attacker_method
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.helper.config.sample_method == 'random':
            file_name = f'{self.helper.config.dataset}/{timestamp}_{self.helper.config.agg_method}_{atk_method}_r_{self.helper.config.num_adversaries}_{self.helper.config.poison_epochs}_ts{self.helper.config.attack_type}.pkl'
        else:
            raise NotImplementedError

        save_path = os.path.join(f'../saved/res/{file_name}')
        directory = os.path.dirname(save_path)
    
        if not os.path.exists(directory):
            os.makedirs(directory)
        f_save = open(save_path, 'wb')
        pickle.dump(log_dict, f_save)
        f_save.close()
        print(f'results saved at {save_path}')


    def train(self):
        
        accs = []
        asrs = []
        self.local_asrs = {}
        for epoch in range(-2, self.helper.config.epochs):
            sampled_participants = self.sample_participants(epoch)
            weight_accumulator, weight_accumulator_by_client = self.train_once(epoch, sampled_participants)
            self.aggregator.agg(self.helper.global_model, weight_accumulator, weight_accumulator_by_client, self.helper.client_models, sampled_participants)
            loss, acc = self.test_once()
            if self.helper.config.agg_method == 'crfl':
                bkd_loss, bkd_acc = self.test_once(poison = self.helper.config.is_poison, logimage = (epoch==self.helper.config.epochs-1),is_log=self.helper.config.is_log)
            else:
                bkd_loss, bkd_acc = self.test_once(poison = self.helper.config.is_poison, logimage = (epoch==self.helper.config.epochs-1),is_log=self.helper.config.is_log)
            # bkd_loss, bkd_acc = self.test_once(poison = self.helper.config.is_poison, logimage = True,is_log=True)
            self.log_once(epoch, loss, acc, bkd_loss, bkd_acc)
            # if self.helper.config.is_poison and epoch % 100 ==0:
            #     self.save_trigger(epoch, self.attacker.trigger, self.helper.global_model)
            accs.append(acc)
            asrs.append(bkd_acc)
        if self.helper.config.is_poison:
            self.save_res(accs, asrs)
            
    def save_trigger(self, epoch, trigger, model):
        if not os.path.exists('../saved/trigger'):
            os.makedirs('../saved/trigger')
        if not os.path.exists('../saved/model'):
            os.makedirs('../saved/model')
        trigger_path = f'../saved/trigger/{self.helper.config.dataset}_{epoch}_{self.helper.config.attack_type}_{self.helper.config.agg_method}.pt'
        model_path = f'../saved/model/{self.helper.config.dataset}_{epoch}_{self.helper.config.attack_type}_{self.helper.config.agg_method}.pt'
        torch.save(trigger, trigger_path)
        torch.save(model, model_path)
        print(f'Trigger saved at {trigger_path}')
        print(f'Model saved at {model_path}')

    def train_once(self, epoch, sampled_participants):
       
        weight_accumulator = self.create_weight_accumulator()
        weight_accumulator_by_client = []
        client_count = 0
        attacker_idxs = []
        global_model_copy = self.create_global_model_copy()
        local_asr = []
        first_adversary = self.contain_adversary(epoch, sampled_participants)
        if first_adversary >= 0 and ('sin' in self.helper.config.attacker_method):
            # if self.attack_sum <13:
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            self.attacker.search_trigger(model, self.helper.train_data[first_adversary], 'outter', first_adversary, epoch)
        if first_adversary >= 0:
            self.attack_sum += 1
            print(f'Epoch {epoch}, poisoning by {first_adversary}, attack sum {self.attack_sum}.')
        else:
            print(f'Epoch {epoch}, no adversary.')

        for participant_id in sampled_participants:
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            model.train()
            if not self.if_adversary(epoch, participant_id, sampled_participants):
                self.train_benign(participant_id, model, epoch)
            else:
                attacker_idxs.append(client_count)
                self.train_malicious(participant_id, model, epoch)

            weight_accumulator, single_wa = self.update_weight_accumulator(model, weight_accumulator)
            weight_accumulator_by_client.append(single_wa)
            self.helper.client_models[participant_id].load_state_dict(model.state_dict())
            client_count += 1
        return weight_accumulator, weight_accumulator_by_client

    def norm_of_update(self, single_wa_by_c, attacker_idxs):
        cossim = torch.nn.CosineSimilarity(dim=0)
        def sim_was(wa1, wa2):
            sim = None
            for name in wa1:
                v1 = wa1[name]
                v2 = wa2[name]
                if v1.dtype == torch.float:
                    sim = cossim(v1.view(-1),v2.view(-1)).item() if sim == None else sim + cossim(v1.view(-1),v2.view(-1)).item()
            return sim
        count = 0
        sim_sum = 0.
        for i in range(len(single_wa_by_c)):
            for j in range(len(single_wa_by_c)):
                if i in attacker_idxs and i != j:
                    sim = sim_was(single_wa_by_c[i], single_wa_by_c[j])
                    sim_sum += sim
                    count += 1
        return sim_sum/count

    def contain_adversary(self, epoch, sampled_participants):
        if self.helper.config.is_poison and \
            epoch < self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'random':
                for p in sampled_participants:
                    if p < self.helper.config.num_adversaries:
                        return p
            elif self.helper.config.sample_method == 'random_updates':
                if epoch in self.advs:
                    return self.advs[epoch][0]
        return -1

    def num_attackers(self, epoch, sampled_participants):
        n = 0
        if self.helper.config.is_poison and \
            epoch < self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'random':
                for p in sampled_participants:
                    if p < self.helper.config.num_adversaries:
                        n += 1
        return n

    def if_adversary(self, epoch, participant_id, sampled_participants):
        if self.helper.config.is_poison and epoch < self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'random' and participant_id < self.helper.config.num_adversaries:
                return True 
            elif self.helper.config.sample_method == 'random_updates':
                if epoch in self.advs:
                    for idx in self.advs[epoch]:
                        if sampled_participants[idx] == participant_id:
                            return True
        else:
            return False

    def create_local_model_copy(self, model):
        model_copy = dict()
        for name, param in model.named_parameters():
            model_copy[name] = model.state_dict()[name].clone().detach().requires_grad_(False)
        return model_copy

    def create_global_model_copy(self):
        global_model_copy = dict()
        for name, param in self.helper.global_model.named_parameters():
            global_model_copy[name] = self.helper.global_model.state_dict()[name].clone().detach().requires_grad_(False)
        return global_model_copy

    def create_weight_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.helper.global_model.state_dict().items():
            ### don't scale tied weights:
            if name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator
    
    def update_weight_accumulator(self, model, weight_accumulator):
        single_weight_accumulator = dict()
        for name, data in model.state_dict().items():
            if name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - self.helper.global_model.state_dict()[name])
            single_weight_accumulator[name] = data - self.helper.global_model.state_dict()[name]
        return weight_accumulator, single_weight_accumulator

    def train_benign(self, participant_id, model, epoch):
        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        
        if self.helper.config.agg_method == 'fedprox':
            global_model_params = [param.clone().detach() for param in model.parameters()]

        for internal_epoch in range(self.helper.config.retrain_times):
            total_loss = 0.0
            
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = self.criterion(output, labels)

                if self.helper.config.agg_method == 'fedprox':
                    for global_param, local_param in zip(global_model_params, model.parameters()):
                        fedprox_loss =self.helper.config.mu / 2 * torch.norm(global_param - local_param)
                        loss += fedprox_loss
                    

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def scale_up(self, model, curren_num_adv):
        clip_rate = 2/curren_num_adv
        for key, value in model.state_dict().items():
            #### don't scale tied weights:
            if  key == 'decoder.weight' or '__'in key:
                continue
            target_value = self.helper.global_model.state_dict()[key]
            new_value = target_value + (value - target_value) * clip_rate

            model.state_dict()[key].copy_(new_value)
        return model

    def train_malicious(self, participant_id, model, epoch):
        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        clean_model = copy.deepcopy(model)
        if self.helper.config.attack_type == 'my':
            poisoned_dataloader = self.attacker.poison_dataloader(self.helper.train_data[participant_id],model=model)
        elif self.helper.config.attack_type == 'a3fl':
            poisoned_dataloader = self.attacker.poison_dataloader(self.helper.train_data[participant_id])

        if self.helper.config.agg_method == 'fedprox':
            global_model_params = [param.clone().detach() for param in model.parameters()]

        for internal_epoch in range(self.helper.config.attacker_retrain_times):
            total_loss = 0.0
            for inputs, labels in poisoned_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                # inputs, labels = self.attacker.poison_input(inputs, labels)
                output = model(inputs)
                loss = self.attacker_criterion(output, labels)

                if self.helper.config.agg_method == 'fedprox':
                    for global_param, local_param in zip(global_model_params, model.parameters()):
                        fedprox_loss =self.helper.config.mu / 2 * torch.norm(global_param - local_param)
                        loss += fedprox_loss

                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                
                optimizer.step()
                
        
    def get_lr(self, epoch):
        if self.helper.config.lr_method == 'exp':
            tmp_epoch = epoch
            if self.helper.config.is_poison and self.helper.config.load_benign_model:
                tmp_epoch += self.helper.config.poison_start_epoch
            lr = self.helper.config.lr * (self.helper.config.gamma**tmp_epoch)
        elif self.helper.config.lr_method == 'linear':
            if self.helper.config.is_poison or epoch > 1900:
                lr = 0.002
            else:
                lr_init = self.helper.config.lr
                target_lr = self.helper.config.target_lr
                #if self.helper.config.dataset == 'cifar10':
                if epoch <= self.helper.config.epochs/2.:
                    lr = epoch*(target_lr - lr_init)/(self.helper.config.epochs/2.-1) + lr_init - (target_lr - lr_init)/(self.helper.config.epochs/2. - 1)
                else:
                    lr = (epoch-self.helper.config.epochs/2)*(-target_lr)/(self.helper.config.epochs/2) + target_lr

                if lr <= 0.002:
                    lr = 0.002
                # else:
                #     raise NotImplementedError
        return lr

    def sample_participants(self, epoch):
        if self.helper.config.sample_method in ['random', 'random_updates']:
            sampled_participants = random.sample(
                range(self.helper.config.num_total_participants), 
                self.helper.config.num_sampled_participants)
        elif self.helper.config.sample_method == 'fix-rate':
            start_index = (epoch * self.helper.config.num_sampled_participants) % self.helper.config.num_total_participants
            sampled_participants = list(range(start_index, start_index+self.helper.config.num_sampled_participants))
        else:
            raise NotImplementedError
        assert len(sampled_participants) == self.helper.config.num_sampled_participants
        return sampled_participants
    
    def copy_params(self, model, target_params_variables):
        for name, layer in model.named_parameters():
            layer.data = copy.deepcopy(target_params_variables[name])
    
    def log_image_table(self,images,clean_image, predicted, labels, probs):
        "Log a wandb.Table with (img, pred, target, scores)"
        # 🐝 Create a wandb Table to log images, labels and predictions to
        # table = wandb.Table(columns=["image","clean" "pred", "target"]+[f"score_{i}" for i in range(43)])
        table = wandb.Table(columns=["image","clean" ,"pred", "target"])
        for img,clean, pred, targ, prob in zip(images.to("cpu"), clean_image.to("cpu"),predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):

            img = img.detach().numpy().transpose(1, 2, 0)  
            img = (img * 0.5 + 0.5)
            img = (img * 255).astype(np.uint8)  
            clean = clean.detach().numpy().transpose(1, 2, 0)
            clean = (clean * 0.5 + 0.5)
            clean = (clean * 255).astype(np.uint8)
            table.add_data(wandb.Image(img),wandb.Image(clean), pred.item(), targ.item())
        wandb.log({"predictions_table":table}, commit=False)

        