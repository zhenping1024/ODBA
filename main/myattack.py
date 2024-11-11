import sys
sys.path.append("../")

import wandb
import argparse
import yaml
import traceback

import torch
import torchvision
import numpy as np
import random

from fl_utils.helper import Helper
from fl_utils.fler import FLer

import os

def setup_wandb(config_path, sweep,project_name):
    with open(config_path, 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)
    if sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name,)
        return sweep_id
    else:
        config = sweep_configuration['parameters']
        d = dict()
        for k in config.keys():
            v = config[k][list(config[k].keys())[0]]
            if type(v) is list:
                d[k] = {'value':v[0]}
            else:
                d[k] = {'value':v}  
        yaml.dump(d, open('./yamls/tmp.yaml','w'))
        wandb.init(config='./yamls/tmp.yaml')
        return None

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    torch.cuda.set_device(2)
    run = wandb.init()
    run_name = f"{wandb.config['dataset']}_{'poison'}{wandb.config['bkd_ratio']}_{'size'}{wandb.config['trigger_size']}_{'opt'}{wandb.config['trigger_outter_epochs']}_{'l_inf'}{wandb.config['l_inf']}"
    run.name = run_name
    set_seed(wandb.config.seed)
    helper = Helper(wandb.config)
    fler = FLer(helper)
    fler.train()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default = './configs/myattack_cifar10.yaml')
    parser.add_argument('--gpu', default = 2)
    parser.add_argument('--sweep',default= True)
    parser.add_argument('--project_name', default = 'ODBA_CIFAR10')
    args = parser.parse_args()
    
    sweep_id = setup_wandb(args.params, args.sweep,args.project_name)
    if args.sweep:
        wandb.agent(sweep_id, function=main, count=1)
    else:
        main()