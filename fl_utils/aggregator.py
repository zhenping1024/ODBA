import sys
sys.path.append("../")
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18
import copy
import os
from sklearn.cluster import DBSCAN
import pdb
import math
from sklearn.metrics import pairwise as smp

def get_total_samples(data_loader):
    total_samples = 0
    for data in data_loader:
        total_samples += data[0].size(0)
    return total_samples
class Aggregator:
    def __init__(self, helper):
        self.helper = helper
        self.Wt = None
        self.krum_client_ids = []
        self.sum_updates =[]

        # 初始化客户端数量和特征维度等参数
        self.n_clients = self.helper.config.num_sampled_participants
        self.n_features = None  # 稍后根据模型参数确定
        self.wv = np.ones(self.n_clients)  # 初始化权重向量
        # 添加历史更新的存储，使用字典来保存每个客户端的累积更新
        self.history_updates = {}  # key: client_id, value: cumulative update vector
            

    def agg(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants):
        if self.helper.config.agg_method == 'avg':
            return self.average_shrink_models(global_model, weight_accumulator)
        elif self.helper.config.agg_method == 'fedprox':
            return self.average_shrink_models(global_model, weight_accumulator)
        elif self.helper.config.agg_method == 'fednova':
            return self.fednova_aggregate_global_model(global_model, weight_accumulator_by_client, sampled_participants)
        elif self.helper.config.agg_method == 'clip':
            self.clip_updates(weight_accumulator)
            return self.average_shrink_models(global_model, weight_accumulator)
        elif self.helper.config.agg_method == 'krum':
            self.krum(global_model,weight_accumulator_by_client,sampled_participants)
            return 
        elif self.helper.config.agg_method == 'multkrum':
            self.multi_krum(global_model,weight_accumulator_by_client,sampled_participants)
            return
        elif self.helper.config.agg_method == 'dp':
            # self.clip_updates(weight_accumulator)
            # self.dp_updates(weight_accumulator)
            # self.clip_updates(weight_accumulator)
            
            return self.dp_avg(global_model, weight_accumulator)
        elif self.helper.config.agg_method == 'median':
            return self.median_shrink_models(global_model,weight_accumulator_by_client)
        elif self.helper.config.agg_method =='rfa':
            # return self.robust_federated_aggregation(global_model, weight_accumulator_by_client)
            return self.geometric_median_update(global_model, weight_accumulator_by_client)
        elif self.helper.config.agg_method == 'rlr':
            return self.robust_lr_aggregate(global_model, weight_accumulator_by_client, sampled_participants)
        elif self.helper.config.agg_method == 'crfl':
            return self.crfl_agg(global_model,weight_accumulator)
        elif self.helper.config.agg_method == 'bulyan':
            return self.bulyan_aggregate_global_model(global_model, weight_accumulator_by_client, sampled_participants)
        elif self.helper.config.agg_method == 'deep':
            return self.deepsight_aggregate_global_model_v2(client_models,sampled_participants,global_model,weight_accumulator_by_client)
        elif self.helper.config.agg_method =='foolsgold':
            return  self.foolsgold_aggregate(global_model, weight_accumulator_by_client, sampled_participants)
        
    # 添加 foolsgold 方法
    def foolsgold_aggregate(self, global_model, weight_accumulator_by_client, sampled_participants):
        """
        Implement FoolsGold defense algorithm for aggregation.
        """
        # 首先，将每个客户端的更新转换为向量形式，并维护历史更新
        client_updates = []
        for idx, client_id in enumerate(sampled_participants):
            client_update = weight_accumulator_by_client[idx]
            update = []
            for name, data in client_update.items():
                if 'num_batches_tracked' in name:
                    continue
                update.append(data.view(-1).cpu().numpy())
            update = np.concatenate(update)
            client_updates.append(update)

            # 更新历史更新
            if client_id in self.history_updates:
                self.history_updates[client_id] += update
            else:
                self.history_updates[client_id] = update.copy()

        client_updates = np.array(client_updates)  # Shape: (n_clients, n_features)

        # 初始化特征维度
        if self.n_features is None:
            self.n_features = client_updates.shape[1]

        # 获取对应的历史更新
        summed_deltas = []
        for client_id in sampled_participants:
            summed_deltas.append(self.history_updates[client_id])
        summed_deltas = np.array(summed_deltas)

        # 获取全局模型的扁平化参数
        global_params = []
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            global_params.append(data.view(-1).cpu().numpy())
        global_params = np.concatenate(global_params)

        # 调用 foolsgold 方法，计算权重向量
        self.wv = self.foolsgold(client_updates, summed_deltas)
        # print(f'[foolsgold agg] wv: {self.wv}')
        # 根据权重向量，更新全局模型
        lr = 1
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            # 收集所有客户端在该参数上的更新
            updates = np.array([weight_accumulator_by_client[idx][name].cpu().numpy() for idx in range(len(sampled_participants))])
            # 计算加权平均的更新
            weighted_update = np.average(updates, axis=0, weights=self.wv)
            weighted_update = weighted_update / len(sampled_participants)
            weighted_update = torch.tensor(weighted_update, dtype=data.dtype)
            data.add_(weighted_update.cuda())

        return True

    # 实现 foolsgold 算法
    def foolsgold(self, this_delta, summed_deltas, epsilon=1e-5):
        """
        FoolsGold defense algorithm.
        """
        n_clients = this_delta.shape[0]

        # 计算客户端历史更新的余弦相似度矩阵
        cs = smp.cosine_similarity(summed_deltas) - np.eye(n_clients)

        # Pardoning：根据最大相似度进行重新加权
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        # 计算权重向量
        wv = 1 - np.max(cs, axis=1)
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # 重新缩放权重
        wv = wv / np.max(wv)
        wv[wv == 1] = 0.99

        # 应用 Logit 函数
        wv = (np.log(wv / (1 - wv) + epsilon) + 0.5)
        wv[np.isinf(wv) | (wv > 1)] = 1
        wv[wv < 0] = 0

        # # 归一化权重
        # wv = wv / np.sum(wv)

        # # 计算加权平均更新
        # avg_updates = np.average(this_delta, axis=0, weights=wv)

        return  wv
 

    def average_shrink_models(self,  global_model, weight_accumulator):

        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        lr = 1

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.helper.config.num_sampled_participants) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return True
    
    def dp_avg(self,global_model,weight_accumulator):
        sigma = 0.002
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.helper.config.num_sampled_participants) 
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())
        for name, param in global_model.state_dict().items():
                    if 'tracked' in name or 'running' in name:
                        continue
                    # print(name)
                    dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                    param.add_(dp_noise)
        
        return

    def crfl_agg(self,global_model,weight_accumulator):

        def get_global_model_norm(global_model):
         squared_sum = 0
         for name, layer in global_model.named_parameters():
             squared_sum += torch.sum(torch.pow(layer.data, 2))
         return math.sqrt(squared_sum)

        def clip_norm(global_model,clip=100):
            total_norm =get_global_model_norm(global_model)
            max_norm = clip
            clip_coef = max_norm / (total_norm + 1e-6)
            current_norm = total_norm
            if total_norm > max_norm:
                for name, layer in global_model.named_parameters():
                    layer.data.mul_(clip_coef)
            
            return
        
        def add_noise(global_model,sigma=0.002, cp=False):
            '''
            向模型权重添加差分隐私噪声
            '''
            if not cp:
                for name, param in global_model.state_dict().items():
                    if 'tracked' in name or 'running' in name:
                        continue
                    # print(name)
                    dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                    param.add_(dp_noise)
            else:
                smoothed_model = copy.deepcopy(global_model)
                for name, param in smoothed_model.state_dict().items():
                    if 'tracked' in name or 'running' in name:
                        continue
                    dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                    param.add_(dp_noise)
            return 


        lr = 1
        clip_norm(global_model)
        add_noise(global_model=global_model)
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.helper.config.num_sampled_participants) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return

    def median_shrink_models(self, global_model, weight_accumulator):

        """
        Perform model aggregation using median
        """
        # 遍历全局模型的每一层参数
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            
            # 获取该层所有客户端的更新，并计算中位数
            layer_updates = torch.stack([participant_update[name] for participant_update in weight_accumulator], dim=0)

            # 计算每一层参数更新的中位数
            median_update_per_layer = torch.median(layer_updates, dim=0)[0]

            # 将中位数更新添加到全局模型的对应参数中
            median_update_per_layer = torch.tensor(median_update_per_layer, dtype=data.dtype)
            data.add_(median_update_per_layer.cuda())  # 将更新后的值应用到全局模型参数中

        return True
    
    def fednova_aggregate_global_model(self, global_model, weight_accumulator_by_client, sampled_participants):
        """
        FedNova algorithm.
        global_model: global model
        weight_accumulator_by_client: 每个客户端的权重更新（全剧模型减去局部模型）
        sampled_participants: 采样的客户端ID
        """
        # 获取全局模型的初始状态
        original_params = global_model.state_dict()

        # 收集每个客户端的数据样本数
        list_num_samples = [get_total_samples(self.helper.train_data[id]) for id in sampled_participants]

        # 计算总样本量
        total_sample = sum(list_num_samples)

        # 动量参数rho，FedNova中通常使用0.9（根据需要可调整）
        rho = 0.9
        tau_list = []

        # 计算每个客户端的tau值，tau代表每个客户端的训练步数
        for client_id in sampled_participants:
            num_samples = get_total_samples(self.helper.train_data[client_id])
            tau = num_samples * 2  # 假设每个样本进行两次训练，这可以根据实际需要调整
            tau_list.append(tau)

        # 初始化全局模型参数的累积更新量
        d_total_round = {key: torch.zeros_like(param) for key, param in original_params.items()}

        # 初始化FedNova中的总系数，用于全局模型参数更新
        total_coeff = 0.0

        # 遍历每个客户端，进行权重更新的归一化和聚合
        for i, client_id in enumerate(sampled_participants):
            # 计算当前客户端的tau值和归一化因子a_i
            tau = tau_list[i]
            a_i = (tau - rho * (1 - pow(rho, tau)) / (1 - rho)) / (1 - rho)

            # 获取客户端的模型更新 (global_model - local_model)
            client_weight_update = weight_accumulator_by_client[i]

            # 归一化客户端的权重更新，并根据数据量加权
            for key in client_weight_update:
                d_total_round[key] += torch.tensor((client_weight_update[key] / a_i) * (list_num_samples[i] / total_sample),dtype=d_total_round[key].dtype)
                # d_total_round[key] += ((client_weight_update[key] / a_i) * (list_num_samples[i] / total_sample)).clone().detach()

            # 计算全局模型更新的加权系数
            total_coeff += a_i * (list_num_samples[i] / total_sample)

        # 使用累积的更新量更新全局模型
        updated_model = global_model.state_dict()
        for key in updated_model:
            # 根据加权的更新值调整全局模型的参数
            if updated_model[key].dtype == torch.int64:
                updated_model[key] += (total_coeff * d_total_round[key]).type(torch.int64)
            else:
                updated_model[key] += total_coeff * d_total_round[key]

        # 将更新后的模型参数加载回全局模型
        global_model.load_state_dict(updated_model)

        return
    
    def dp_updates(self,weight_accumulator):
        for name, weights in weight_accumulator.items():
        # Ensure the weights are on the same device as the noise
            device = weights.device
            noise = torch.normal(0, 0.002, size=weights.size(), device=device)
            # print("weight=",weights)
            # print("noise=",noise)
            weights = weights + noise
             
        return
    
    def clip_updates(self, agent_updates_dict):
        for key in agent_updates_dict:
            if 'num_batches_tracked' not in key:
                update = agent_updates_dict[key]
                l2_update = torch.norm(update, p=2) 
                update.div_(max(1, l2_update/self.helper.config.clip_factor))
        return
    
    def robust_lr_aggregate(self, global_model, weight_accumulator_by_client, sampled_participants):
        """
        Perform robust learning rate aggregation using sign voting method.
        """
        original_params = global_model.state_dict()
        total_sample = sum([len(self.helper.train_data[id].dataset)for id in sampled_participants])

        # 收集客户端更新
        updates = weight_accumulator_by_client
  

        # 计算鲁棒学习率
        robust_lrs = self.compute_robust_lr(updates)

        # 进行符号投票聚合
        flip_analysis = {}
        for layer in robust_lrs.keys():
            n_flip = torch.sum(torch.gt(robust_lrs[layer], 0.0).int())
            n_unflip = torch.sum(torch.lt(robust_lrs[layer], 0.0).int())
            flip_analysis[layer] = [n_flip, n_unflip]

        # 根据符号投票结果更新全局模型
        for i, id in enumerate(sampled_participants):
            client_update = weight_accumulator_by_client[i]
            prop = len(self.helper.train_data[id].dataset) / total_sample
            for layer in original_params.keys():
                if layer == 'decoder.weight':
                    continue
            # if 'running' in layer or 'tracked' in layer:
            #     tmp_updates=update[layer]*prop
            #     tmp_updates =torch.tensor(tmp_updates,dtype=original_params[layer].dtype)
            #     original_params[layer] += tmp_updates
            # else:
                tmp_updates=client_update[layer]*prop*robust_lrs[layer]
                tmp_updates =torch.tensor(tmp_updates,dtype=original_params[layer].dtype)
                original_params[layer] += tmp_updates
            # self.robust_lr_add_weights(original_params, robust_lrs, client_update, prop)

        global_model.load_state_dict(original_params)
        return 

    def compute_robust_lr(self, updates):
        """
        Compute the robust learning rates based on client updates.
        """
        layers = updates[0].keys()
        robust_lrs = OrderedDict()
        for layer in layers:
            robust_lrs[layer] = torch.zeros_like(updates[0][layer])

        for layer in layers:
            for update in updates:
                robust_lrs[layer] += torch.sign(update[layer])
            robust_lrs[layer] = torch.abs(robust_lrs[layer])
            robust_lrs[layer][robust_lrs[layer] >= 2] = 1.0
            robust_lrs[layer][robust_lrs[layer] != 1.0] = -1.0
        return robust_lrs

    def robust_lr_add_weights(self, original_params, robust_lrs, update, prop):
        """
        Update global model weights using robust learning rate and client updates.
        """
        # for name, data in original_params.state_dict().items():
        #     if name == 'decoder.weight':
        #         continue
        #     update_per_layer = update[name] * prop*robust_lrs[name]
        #     update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
        #     data.add_(update_per_layer.cuda())
        for layer in original_params.keys():
            if layer == 'decoder.weight':
                continue
            # if 'running' in layer or 'tracked' in layer:
            #     tmp_updates=update[layer]*prop
            #     tmp_updates =torch.tensor(tmp_updates,dtype=original_params[layer].dtype)
            #     original_params[layer] += tmp_updates
            # else:
            tmp_updates=update[layer]*prop*robust_lrs[layer]
            tmp_updates =torch.tensor(tmp_updates,dtype=original_params[layer].dtype)
            original_params[layer] += tmp_updates

    def bulyan_aggregate_global_model(self, global_model, weight_accumulator_by_client, sampled_participants):
        """
        实现 Bulyan 聚合方法，并使用与 average_shrink_models 一致的方式更新全局模型。

        参数:
        - global_model: 全局模型
        - weight_accumulator_by_client: 每个客户端的更新权重（字典列表）
        - sampled_participants: 当前采样的客户端列表
        """
        num_clients = len(sampled_participants)
        num_adv = self.helper.config.num_adversaries  # 恶意客户端数量
        f = num_adv  # 假设拜占庭客户端数量为恶意客户端数量

        theta = num_clients - 2 * f  # 需要选择的更新数量

        selected_updates = []
        remaining_updates = weight_accumulator_by_client.copy()
        remaining_indices = list(range(len(weight_accumulator_by_client)))
        
        # Bulyan第一阶段：迭代使用Krum算法选择theta个更新
        for _ in range(theta):
            num_remaining = len(remaining_updates)
            nb_in_score = num_remaining - f - 2
            if nb_in_score < 1:
                nb_in_score = 1  # 确保至少有一个邻居

            # 将剩余更新转换为向量形式
            vectorized_updates = []
            for client_model in remaining_updates:
                updates = []
                for layer_update in client_model.values():
                    updates.append(torch.tensor(layer_update).flatten())
                vectorized_updates.append(torch.cat(updates).cpu().detach().numpy())

            # 计算剩余更新之间的欧氏距离
            distances = np.zeros((num_remaining, num_remaining))
            for i in range(num_remaining):
                for j in range(i + 1, num_remaining):
                    distances[i, j] = np.linalg.norm(vectorized_updates[i] - vectorized_updates[j]) ** 2
                    distances[j, i] = distances[i, j]

            # 计算每个更新的得分
            scores = []
            for i in range(num_remaining):
                dists = np.sort(distances[i])
                scores.append(np.sum(dists[:nb_in_score + 1]))  # 包含自身距离（为0）

            # 选择得分最低的更新
            i_star = np.argmin(scores)
            selected_updates.append(remaining_updates[i_star])

            # 从剩余更新中移除已选中的更新
            del remaining_updates[i_star]
            del remaining_indices[i_star]


        original_params = global_model.state_dict()
        # print("Bulyan Stage 2：", len(krum_updates))    
        bulyan_update = OrderedDict()
        layers = selected_updates[0].keys()
        for layer in layers:
            bulyan_layer = None
            for update in selected_updates:
                bulyan_layer = update[layer][None, ...] if bulyan_layer is None else torch.cat(
                    (bulyan_layer, update[layer][None, ...]), 0)

            med, _ = torch.median(bulyan_layer, 0)
            _, idxs = torch.sort(torch.abs(bulyan_layer - med), 0)
            bulyan_layer = torch.gather(bulyan_layer, 0, idxs[:-2*f, ...])
            # print("bulyan_layer",bulyan_layer.size())
            # bulyan_update[layer] = torch.mean(bulyan_layer, 0)
            # print(bulyan_layer)
            if not 'tracked' in layer:
                bulyan_update[layer] = torch.mean(bulyan_layer, 0)
            else:
                bulyan_update[layer] = torch.mean(bulyan_layer*1.0, 0).long()
            original_params[layer] = original_params[layer] + bulyan_update[layer]

        global_model.load_state_dict(original_params)
        
        return

    def compute_pairwise_distance(self, updates):
        '''
        计算客户端之间的权重更新的欧式距离
        '''
        def pairwise(u1, u2):
            ks = u1.keys()
            dist = 0
            for k in ks:
                if 'tracked' in k:
                    continue
                d = u1[k] - u2[k]
                dist = dist + torch.sum(d * d)
            return round(float(torch.sqrt(dist)), 2)

        scores = [0 for u in range(len(updates))]
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                dist = pairwise(updates[i], updates[j])
                scores[i] = scores[i] + dist
                scores[j] = scores[j] + dist
        return scores

    def deepsight_aggregate_global_model(self, global_model,  weight_accumulator_by_client,client_models,sampled_participants):

        '''
        使用DeepSight算法聚合全局模型
        '''
        def ensemble_cluster(neups, ddifs, biases):
            biases = np.array([bias.cpu().numpy() for bias in biases])
            N = len(neups)
            # 使用DBSCAN对偏置进行聚类
            cosine_labels = DBSCAN(min_samples=3, metric='cosine').fit(biases).labels_
            # print("cosine_cluster:{}".format(cosine_labels))

            neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
            # print("neup_cluster:{}".format(neup_labels))

            ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
            # print("ddif_cluster:{}".format(ddif_labels))

            # 计算各聚类的距离
            dists_from_cluster = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                        neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j])) / 3.0
                    dists_from_cluster[j, i] = dists_from_cluster[i, j]

            # print("dists_from_clusters:")
            # print(dists_from_cluster)

            ensembled_labels = DBSCAN(min_samples=3, metric='precomputed').fit(dists_from_cluster).labels_

            return ensembled_labels
        # print(sampled_participants,'...................')
        # 获取全局模型的权重和偏置
        global_weight = list(global_model.state_dict().values())[-2]
        global_bias = list(global_model.state_dict().values())[-1]

        # 收集每个客户端的偏置和权重
        # biases = [(list(weight_accumulator_by_client[i].values())[-1]) for i in range(len(sampled_participants))]
        # weights = [list(weight_accumulator_by_client[i].values())[-2] for i in range(len(sampled_participants))]
        biases =[(list(client_models[i].state_dict().values())[-1] - global_bias) for i in sampled_participants]
        weights = [list(client_models[i].state_dict().values())[-2] for i in sampled_participants]
        
        n_client = len(sampled_participants)
        cosine_similarity_dists = np.zeros((n_client, n_client))  # 初始化相似度矩阵
        neups = list()
        n_exceeds = list()

        # 计算neups
        sC_nn2 = 0
        for i in range(n_client):
            C_nn = torch.sum(weights[i] - global_weight, dim=[1]) + biases[i] - global_bias
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2

            C_max = torch.max(C_nn2).item()
            threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else (1 / len(biases)) * C_max
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)

        # 归一化neups
        neups = np.array([(neup / sC_nn2).cpu().numpy() for neup in neups])
        # print("n_exceeds:{}".format(n_exceeds))

        # 随机输入生成
        rand_input = None
        if self.helper.config.dataset == 'cifar10' or self.helper.config.dataset == 'GTSRB':
            rand_input = torch.randn((256, 3, 32, 32)).cuda()
        elif self.helper.config.dataset == 'tiny-imagenet':
            rand_input = torch.randn((128, 3, 64, 64)).cuda()

        # 计算全局模型和客户端模型的ddif
        global_ddif = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0)
        client_ddifs = [torch.mean(torch.softmax(self.helper.client_models[i](rand_input), dim=1), dim=0) / global_ddif
                        for i in sampled_participants]
        client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])

        # 使用n_exceed来识别恶意客户端
        classification_boundary = np.median(np.array(n_exceeds)) / 2
        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
        # print("identified_mals:{}".format(identified_mals))

        # 使用ensemble_cluster进行聚类
        clusters = ensemble_cluster(neups, client_ddifs, biases)
        # print("ensemble clusters:{}".format(clusters))
        cluster_ids = np.unique(clusters)

        deleted_cluster_ids = list()
        for cluster_id in cluster_ids:
            n_mal = 0
            cluster_size = np.sum(cluster_id == clusters)
            for identified_mal, cluster in zip(identified_mals, clusters):
                if cluster == cluster_id and identified_mal:
                    n_mal += 1
            # print("cluster size:{} n_mal:{}".format(cluster_size, n_mal))
            if (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)

        temp_sampled_participants = copy.deepcopy(sampled_participants)
        for i in range(len(sampled_participants) - 1, -1, -1):
            if clusters[i] in deleted_cluster_ids:
                del sampled_participants[i]

        print("final clients length:{}".format(len(sampled_participants)))
        if len(sampled_participants) == 0:
            sampled_participants = temp_sampled_participants

        self.deepsight_average(global_model, client_models,sampled_participants)
    
    def deepsight_average(self,  global_model, client_models,chosen_id):

        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        # print(chosen_id,',,,,,,,,,,,,,,,,,,,,,,,,')
        weight_accumulator = self.create_weight_accumulator()
        for i in range(len(chosen_id)):
            for name, data in client_models[chosen_id[i]].state_dict().items():
                if name == 'decoder.weight' or '__'in name:
                    continue
                weight_accumulator[name].add_(data - global_model.state_dict()[name])
        lr = 1

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * (1/len(chosen_id)) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return True
    def create_weight_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.helper.global_model.state_dict().items():
            ### don't scale tied weights:
            if name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator
    def krum(self, global_model, weight_accumulator_by_client, sampled_participants):
        """
        实现 Krum 聚合方法，并使用与 average_shrink_models 一致的方式更新全局模型。

        参数:
        - global_model: 全局模型的参数
        - weight_accumulator_by_client: 每个客户端的更新权重
        - sampled_participants: 当前采样的客户端列表
        """
        # vectorized_updates = [torch.nn.utils.parameters_to_vector(client_model)
        #                       for client_model in weight_accumulator_by_client]
        # 将每个客户端的更新转化为向量
        vectorized_updates = []
        for client_model in weight_accumulator_by_client:
            # 如果 weight_accumulator_by_client 是字典类型，提取每个层的更新并转化为向量
            updates = []
            for layer_update in client_model.values():
                updates.append(torch.tensor(layer_update).flatten())  # 将每个层的更新展平成向量
            vectorized_updates.append(torch.cat(updates).cpu().detach().numpy())  # 将各层的向量合并成一个整体

        num_clients = len(sampled_participants)
        num_adv = self.helper.config.num_adversaries
        nb_in_score = num_clients - num_adv - 2

        # 计算每对客户端之间的欧几里得距离，并存储距离矩阵
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                distances[i, j] = np.linalg.norm(vectorized_updates[i] - vectorized_updates[j]) ** 2
                distances[j, i] = distances[i, j]  # 对称距离矩阵

        # 计算每个客户端的得分
        scores = []
        for i in range(num_clients):
            dists = np.sort(distances[i])  # 按距离排序
            scores.append(np.sum(dists[:nb_in_score + 1]))  # 计算前 nb_in_score 个邻居的距离总和

        # 选择得分最低的客户端
        i_star = np.argmin(scores)

        # 使用与 average_shrink_models 类似的方式更新模型
        lr = 1
        for name, data in global_model.state_dict().items():
            if name in weight_accumulator_by_client[i_star]:  # 仅处理有效的更新
                update_per_layer = weight_accumulator_by_client[i_star][name]
                update_per_layer = torch.tensor(update_per_layer, dtype=data.dtype)
                data.add_(update_per_layer.cuda())

        return 

    def multi_krum(self, global_model, weight_accumulator_by_client, sampled_participants):

        """
        实现 Multi-Krum 聚合方法，并使用与 average_shrink_models 一致的方式更新全局模型。

        参数:
        - global_model: 全局模型的参数
        - weight_accumulator_by_client: 每个客户端的更新权重
        - sampled_participants: 当前采样的客户端列表
        """
        # vectorized_updates = [torch.nn.utils.parameters_to_vector(client_model).cpu().detach().numpy()
        #                       for client_model in weight_accumulator_by_client]
        vectorized_updates = []
        for client_model in weight_accumulator_by_client:
            # 如果 weight_accumulator_by_client 是字典类型，提取每个层的更新并转化为向量
            updates = []
            for layer_update in client_model.values():
                updates.append(torch.tensor(layer_update).flatten())  # 将每个层的更新展平成向量
            vectorized_updates.append(torch.cat(updates).cpu().detach().numpy())  # 将各层的向量合并成一个整体

        num_clients = len(sampled_participants)
        num_adv = self.helper.config.num_adversaries
        nb_in_score = num_clients - num_adv - 2

        # 计算客户端之间的距离矩阵
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                distances[i, j] = np.linalg.norm(vectorized_updates[i] - vectorized_updates[j]) ** 2
                distances[j, i] = distances[i, j]

        # 计算每个客户端的得分
        scores = []
        for i in range(num_clients):
            dists = np.sort(distances[i])
            scores.append(np.sum(dists[:nb_in_score + 1]))

        # 选取得分最低的 num_clients - num_adv 个客户端
        selected_clients = np.argpartition(scores, num_clients - num_adv)[:num_clients - num_adv]

        # 对选中的客户端更新进行聚合
        lr = 1
        for name, data in global_model.state_dict().items():
            aggregated_update = sum(weight_accumulator_by_client[i][name] for i in selected_clients) / len(selected_clients)
            aggregated_update = torch.tensor(aggregated_update, dtype=data.dtype)
            data.add_(aggregated_update.cuda())

        return 
    

    def robust_federated_aggregation(self, global_model, weight_accumulator_by_client):
        """
        使用几何中位数执行鲁棒联邦聚合（RFA）。
        """
        import numpy as np
        client_updates = []

        # 收集并展平客户端更新
        for client_update_dict in weight_accumulator_by_client:
            client_update = []
            for name, param in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                update = client_update_dict[name]
                update = update.cpu().numpy().flatten()
                client_update.append(update)
            client_update = np.concatenate(client_update)
            client_updates.append(client_update)

        client_updates = np.array(client_updates)  # 形状: (num_clients, total_params)

        # 计算几何中位数
        median_update = self.geometric_median(client_updates)

        # 更新全局模型
        lr = 1  # 学习率
        idx = 0
        for name, param in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            param_shape = param.size()
            param_size = param.numel()
            median_param_update = median_update[idx:idx + param_size]
            median_param_update = median_param_update.reshape(param_shape)
            median_param_update = torch.tensor(median_param_update, dtype=param.dtype)
            param.data.add_(median_param_update.to(param.device) * lr)
            idx += param_size

        return True

    def geometric_median(self, points, max_iter=4, eps=1e-5):
        """
        使用Weiszfeld算法计算一组点的几何中位数。
        """
        import numpy as np

        median = np.mean(points, axis=0)
        for _ in range(max_iter):
            prev_median = median.copy()
            distances = np.linalg.norm(points - median, axis=1)
            # 避免除以零
            near_zero = distances < eps
            if np.any(near_zero):
                # 如果任何距离为零，将中位数设置为该点
                median = points[near_zero][0]
                break
            inv_distances = 1.0 / distances
            weights = inv_distances / np.sum(inv_distances)
            median = np.sum(weights[:, np.newaxis] * points, axis=0)
            # 检查收敛性
            if np.linalg.norm(median - prev_median) < eps:
                break
        return median
    
    # 几何中位数更新，用于抗攻击聚合
    def geometric_median_update(self, target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm=None):
        points = []
        alphas = [0.1] * len(updates)
        # for name, data in updates.items():
        #     points.append(data[1])  # 收集每个客户端的更新
        #     alphas.append(data[0])  # 收集每个客户端的样本数
        points = updates
        # adver_ratio = 0


        # adver_ratio = adver_ratio / sum(alphas)
        # poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']

        # logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        alphas = torch.from_numpy(alphas).float()

        median = Aggregator.weighted_average_oracle(points, alphas)  # 计算加权平均值
        num_oracle_calls = 1

        obj_val = Aggregator.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        # logger.info(f'[rfa agg] init. name: {names}, weight: {alphas}')

        wv = None
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, Aggregator.l2dist(median, p)) for alpha, p in zip(alphas, points)], dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = Aggregator.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = Aggregator.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val, (prev_obj_val - obj_val) / obj_val, Aggregator.l2dist(median, prev_median)]
            logs.append(log_entry)

            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
            wv = copy.deepcopy(weights)
        alphas = [Aggregator.l2dist(median, p) for p in points]

        update_norm = 0
        for name, data in median.items():
            update_norm += torch.sum(torch.pow(data, 2))
        update_norm = math.sqrt(update_norm)

        if max_update_norm is None or update_norm < max_update_norm:  # 如果更新的范数不过大，则应用更新
            for name, data in target_model.state_dict().items():
                update_per_layer = median[name]
                data.add_(update_per_layer)
            is_updated = True
        else:
            is_updated = False

        return 

    @staticmethod
    def l2dist(p1, p2):
        """计算p1和p2之间的L2距离"""
        squared_sum = 0
        for name, data in p1.items():
            squared_sum += torch.sum(torch.pow(p1[name] - p2[name], 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """计算几何中位数的目标值"""
        temp_sum = 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * Aggregator.l2dist(median, p)
        return temp_sum

    @staticmethod
    def weighted_average_oracle(points, weights):
        """计算加权平均"""
        tot_weights = torch.sum(weights)

        weighted_updates = dict()

        for name, data in points[0].items():
            weighted_updates[name] = torch.zeros_like(data)
        for w, p in zip(weights, points):  # 对每一个agent
            for name, data in weighted_updates.items():
                temp = (w / tot_weights).float()
                temp = temp * (p[name].float())
                if temp.dtype != data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates

    def deepsight_aggregate_global_model_v2(self, clients, chosen_ids,global_model,weight_accumulator_by_client):
        '''
        使用DeepSight算法聚合全局模型
        '''
        def ensemble_cluster(neups, ddifs, biases):
            biases = np.array([bias.cpu().numpy() for bias in biases])
            #neups = np.array([neup.cpu().numpy() for neup in neups])
            #ddifs = np.array([ddif.cpu().detach().numpy() for ddif in ddifs])
            N = len(neups)
            # use bias to conduct DBSCAM
            # biases= np.array(biases)
            cosine_labels = DBSCAN(min_samples=3,metric='cosine').fit(biases).labels_
            # print("cosine_cluster:{}".format(cosine_labels))
            # neups=np.array(neups)
            neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
            # print("neup_cluster:{}".format(neup_labels))
            ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
            # print("ddif_cluster:{}".format(ddif_labels))

            dists_from_cluster = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                        neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j]))/3.0
                    dists_from_cluster[j, i] = dists_from_cluster[i, j]
                    
            # print("dists_from_clusters:")
            # print(dists_from_cluster)
            ensembled_labels = DBSCAN(min_samples=3,metric='precomputed').fit(dists_from_cluster).labels_

            return ensembled_labels
        
        global_weight = list(global_model.state_dict().values())[-2]
        global_bias = list(global_model.state_dict().values())[-1]

        biases = [(list(clients[i].state_dict().values())[-1] - global_bias) for i in chosen_ids]
        weights = [list(clients[i].state_dict().values())[-2] for i in chosen_ids]

        n_client = len(chosen_ids)
        cosine_similarity_dists = np.array((n_client, n_client))
        neups = list()
        n_exceeds = list()

        # calculate neups
        sC_nn2 = 0
        for i in range(len(chosen_ids)):
            C_nn = torch.sum(weights[i]-global_weight, dim=[1]) + biases[i]-global_bias
            # print("C_nn:",C_nn)
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2
            
            C_max = torch.max(C_nn2).item()
            threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)
        # normalize
        neups = np.array([(neup/sC_nn2).cpu().numpy() for neup in neups])
        # print("n_exceeds:{}".format(n_exceeds))
        rand_input = None
        if self.helper.config.dataset == 'cifar10' or self.helper.config.dataset == 'GTSRB':
            rand_input = torch.randn((256, 3, 32, 32)).cuda()
        elif self.helper.config.dataset == 'tiny-imagenet':
            rand_input = torch.randn((128, 3, 64, 64)).cuda()

        global_ddif = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0)
        # print("global_ddif:{} {}".format(global_ddif.size(),global_ddif))
        client_ddifs = [torch.mean(torch.softmax(clients[i](rand_input), dim=1), dim=0)/ global_ddif
                        for i in chosen_ids]
        client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])
        # print("client_ddifs:{}".format(client_ddifs[0]))

        # use n_exceed to label
        classification_boundary = np.median(np.array(n_exceeds)) / 2
        
        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
        # print("identified_mals:{}".format(identified_mals))
        clusters = ensemble_cluster(neups, client_ddifs, biases)
        # print("ensemble clusters:{}".format(clusters))
        cluster_ids = np.unique(clusters)

        deleted_cluster_ids = list()
        for cluster_id in cluster_ids:
            n_mal = 0
            cluster_size = np.sum(cluster_id == clusters)
            for identified_mal, cluster in zip(identified_mals, clusters):
                if cluster == cluster_id and identified_mal:
                    n_mal += 1
            # print("cluser size:{} n_mal:{}".format(cluster_size,n_mal))        
            if (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)
        # print("deleted_clusters:",deleted_cluster_ids)
        temp_chosen_ids = copy.deepcopy(chosen_ids)
        for i in range(len(chosen_ids)-1, -1, -1):
            # print("cluster tag:",clusters[i])
            if clusters[i] in deleted_cluster_ids:
                del chosen_ids[i]

        print("final clients length:{}".format(len(chosen_ids)))
        if len(chosen_ids)==0:
            chosen_ids = temp_chosen_ids
        
        for i in range(len(temp_chosen_ids)):
            if temp_chosen_ids[i] in chosen_ids:
                for name, data in global_model.state_dict().items():
                    if name == 'decoder.weight':
                        continue
                    update_per_layer = weight_accumulator_by_client[i][name] * \
                                    (1/len(chosen_ids)) 
                    update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
                    data.add_(update_per_layer.cuda())


        