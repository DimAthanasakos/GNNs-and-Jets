#!/usr/bin/env python3

"""
Train GNNs to classify jets
"""

import os
import sys
import yaml
import pickle
from collections import defaultdict

import torch

sys.path.append('.')
from base import common_base
import data_IO
from analysis.models import gnn_pytorch, particle_net, particle_transformer, nsub_transformer


################################################################
class MLAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, config_file='', output_dir='', gpu_mode='single', **kwargs):
        super(common_base.CommonBase, self).__init__(**kwargs)
        
        self.config_file = config_file
        self.output_dir = output_dir
        self.gpu_mode = gpu_mode
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize config file
        self.initialize_config()
            
        # Set torch device
        os.environ['TORCH'] = torch.__version__
        self.rank = int(os.getenv("LOCAL_RANK", "0"))
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.rank == 0:
            print()
            print(f'pytorch version: {torch.__version__}')
            print('Using device:', self.torch_device)
            if self.torch_device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
            print()
            print(self)
            print()
        
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)
        
        # Which Classification task 
        self.classification_task = config['classification_task']

        self.n_train = config['n_train']
        self.n_val = config['n_val']
        self.n_test = config['n_test']
        self.n_total = self.n_train + self.n_val + self.n_test
        self.test_frac = 1. * self.n_test / self.n_total
        self.val_frac  = 1. * self.n_val /  self.n_total

        # Initialize model-specific settings
        self.models = config['models']
        self.model_settings = {}
        for model in self.models:
            self.model_settings[model] = config[model]
            
    #---------------------------------------------------------------
    # Train models
    #---------------------------------------------------------------
    def train_models(self):
    
        self.AUC = defaultdict(list)
        self.roc_curve_dict = self.recursive_defaultdict()
        for model in self.models:
            if self.rank == 0:
                print()
                print(f'------------- Training model: {model} -------------')
            model_settings = self.model_settings[model]
            model_info = {'model': model,
                          'model_settings': model_settings,
                          'classification_task': self.classification_task, 
                          'n_total': self.n_total,
                          'n_train': self.n_train,
                          'n_val': self.n_val,
                          'n_test': self.n_test,
                          'torch_device': self.torch_device,
                          'output_dir': self.output_dir,
                          'gpu_mode': self.gpu_mode}

            if model in ['particle_net', 'particle_net_laman']: 
                model_key = f'{model}'
                print(f'model_key: {model_key}')
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                self.AUC[model_key], self.roc_curve_dict[model_key] = particle_net.ParticleNet(model_info_temp).train()

            if model in ['particle_transformer', 'particle_transformer_graph']:
                model_key = f'{model}'
                if self.rank == 0:
                    print(f'model_key: {model_key}')
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                self.AUC[model_key], self.roc_curve_dict[model_key] = particle_transformer.ParT(model_info_temp).train()

            if model in ['nsub_transformer']:
                model_key = f'{model}'
                if self.rank == 0:
                    print(f'model_key: {model_key}')
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                nsub_transformer.nsubTrans(model_info_temp).train()
                print('DONE NICE')
                

            # TODO: PFN (tensorflow)

            # TODO: PFN (pytorch)

            # ---------- Write ROC curve dict to file ----------
            output_filename = os.path.join(self.output_dir, f'ROC.pkl')
            with open(output_filename, 'wb') as f:
                pickle.dump(self.roc_curve_dict, f)
                pickle.dump(self.AUC, f)