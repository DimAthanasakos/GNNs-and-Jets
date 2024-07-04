#!/usr/bin/env python3

"""
Train GNNs to classify jets
"""

import os
import sys
import yaml
import pickle
from collections import defaultdict
import numpy as np
import torch

sys.path.append('.')
from base import common_base
import data_IO
from analysis.models import gnn_pytorch, particle_net, particle_transformer, nsub_trans, nsub_dnn, subjet_transformer, subjet_nsub_dnn, subjet_nsub_trans, efp


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
            
            if model=='efp':
                model_key = f'{model}'
                print(f'model_key: {model_key}')
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                self.AUC[model_key], self.roc_curve_dict[model_key] = efp.efpDNN(model_info_temp).train()

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
                trim_particles_list = model_info_temp['model_settings']['trim_particles']
                for trim_particles in trim_particles_list:
                    model_info_temp['model_settings']['trim_particles'] = trim_particles
                    self.AUC[model_key], self.roc_curve_dict[model_key] = particle_transformer.ParT(model_info_temp).train()


            if model in ['subjet_transformer', 'subjet_transformer_graph']:
                model_key = f'{model}'
                if self.rank == 0:
                    print(f'model_key: {model_key}')
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                trim_particles_list = model_info_temp['model_settings']['trim_particles']
                cluster_list = model_info_temp['model_settings']['cluster_list']
                for N_cluster in cluster_list:
                    model_info_temp['model_settings']['N_cluster'] = N_cluster
                    for trim_particles in trim_particles_list:
                        model_info_temp['model_settings']['trim_particles'] = trim_particles
                        self.AUC[model_key], self.roc_curve_dict[model_key] = subjet_transformer.ParT(model_info_temp).train()


            if model in ['nsub_dnn']:
                model_key = f'{model}'
                if self.rank == 0:
                    print(f'model_key: {model_key}')
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                k_list = model_info_temp['model_settings']['K']
                for k in k_list:
                    model_info_temp['model_settings']['K'] = k
                    self.AUC[model_key], self.roc_curve_dict[model_key]  = nsub_dnn.nsubDNN(model_info_temp).train()
            

            if model in ['subjet_nsub_dnn']:
                model_key = f'{model}'
                if self.rank == 0:
                    print(f'model_key: {model_key}')
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                k_list = model_info_temp['model_settings']['K']
                cluster_list = model_info_temp['model_settings']['cluster_list']
                runs = model_info_temp['model_settings']['runs']
                auc_list = []
                auc_av_list = []
                auc_std_list = []
                e_05_list = []
                e_08_list = []
                e_05_av_list = []
                e_08_av_list = []
                e_05_std_list = []
                e_08_std_list = []
                for N_cluster in cluster_list:
                    model_info_temp['model_settings']['N_cluster'] = N_cluster
                    auc_cur_list, e05_list, e08_list = [], [], []

                    for run in range(runs):
                        for k in k_list:
                            model_info_temp['model_settings']['K'] = k
                            self.AUC[model_key], self.roc_curve_dict[model_key], e_05, e_08  = subjet_nsub_dnn.nsubDNN(model_info_temp).train()
                            auc_cur_list.append(self.AUC[model_key])
                            e05_list.append(e_05)
                            e08_list.append(e_08)
                    auc_av_list.append(sum(auc_cur_list)/len(auc_cur_list))
                    auc_std_list.append(np.std(auc_cur_list) if len(auc_cur_list) > 1 else 0)
                    auc_list.append(auc_cur_list) 
                    e_05_av_list.append(sum(e05_list)/len(e05_list))
                    e_05_std_list.append(np.std(e05_list) if len(e05_list) > 1 else 0)
                    e_05_list.append(e05_list)
                    e_08_av_list.append(sum(e08_list)/len(e08_list))
                    e_08_std_list.append(np.std(e08_list) if len(e08_list) > 1 else 0)
                    e_08_list.append(e08_list)
                print('===============================================')
                print('AUC list: ', auc_list)
                print(f'AUC average: {auc_av_list}')
                print(f'AUC std: {auc_std_list}')
                print()
                print(f'e_05 list: {e_05_list}')
                print(f'e_05 average: {e_05_av_list}')
                print(f'e_05 std: {e_05_std_list}')
                print()
                print(f'e_08 list: {e_08_list}')
                print(f'e_08 average: {e_08_av_list}')
                print(f'e_08 std: {e_08_std_list}')
                print('===============================================')

                    
                
            if model in ['nsub_transformer']:
                model_key = f'{model}'
                if self.rank == 0:
                    print(f'model_key: {model_key}')
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                k_list = model_info_temp['model_settings']['K']
                for k in k_list:
                    model_info_temp['model_settings']['K'] = k
                    self.AUC[model_key], self.roc_curve_dict[model_key]  = nsub_trans.nsubTrans(model_info_temp).train()

            if model in ['subjet_nsub_transformer']:
                model_key = f'{model}'
                if self.rank == 0: print(f'model_key: {model_key}')
                model_info_temp = model_info.copy()
                model_info_temp['model_key'] = model_key
                k_list = model_info_temp['model_settings']['K']
                cluster_list = model_info_temp['model_settings']['cluster_list']
                runs = model_info_temp['model_settings']['runs']
                auc_list = []
                auc_av_list = []
                auc_std_list = []
                e_05_list = []
                e_08_list = []
                e_05_av_list = []
                e_08_av_list = []
                e_05_std_list = []
                e_08_std_list = []
                for N_cluster in cluster_list:
                    model_info_temp['model_settings']['N_cluster'] = N_cluster
                    auc_cur_list, e05_list, e08_list = [], [], []
                    for run in range(runs):
                        for k in k_list:
                            model_info_temp['model_settings']['K'] = k
                            self.AUC[model_key], self.roc_curve_dict[model_key], e_05, e_08   = subjet_nsub_trans.nsubTrans(model_info_temp).train()                        
                            auc_cur_list.append(self.AUC[model_key])
                            e05_list.append(e_05)
                            e08_list.append(e_08)
                    auc_av_list.append(sum(auc_cur_list)/len(auc_cur_list))
                    auc_std_list.append(np.std(auc_cur_list) if len(auc_cur_list) > 1 else 0)
                    auc_list.append(auc_cur_list) 
                    e_05_av_list.append(sum(e05_list)/len(e05_list))
                    e_05_std_list.append(np.std(e05_list) if len(e05_list) > 1 else 0)
                    e_05_list.append(e05_list)
                    e_08_av_list.append(sum(e08_list)/len(e08_list))
                    e_08_std_list.append(np.std(e08_list) if len(e08_list) > 1 else 0)
                    e_08_list.append(e08_list)
                print('===============================================')
                print('AUC list: ', auc_list)
                print(f'AUC average: {auc_av_list}')
                print(f'AUC std: {auc_std_list}')
                print()
                print(f'e_05 list: {e_05_list}')
                print(f'e_05 average: {e_05_av_list}')
                print(f'e_05 std: {e_05_std_list}')
                print()
                print(f'e_08 list: {e_08_list}')
                print(f'e_08 average: {e_08_av_list}')
                print(f'e_08 std: {e_08_std_list}')
                print('===============================================')

            # TODO: PFN 


            # ---------- Write ROC curve dict to file ----------
            output_filename = os.path.join(self.output_dir, f'ROC.pkl')
            with open(output_filename, 'wb') as f:
                pickle.dump(self.roc_curve_dict, f)
                pickle.dump(self.AUC, f)