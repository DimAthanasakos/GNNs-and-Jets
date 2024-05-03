import os
import time
import numpy as np
import math 
import sys
import glob
from collections import defaultdict


import functools

import socket 

import matplotlib.pyplot as plt
import sklearn
import scipy
from scipy.sparse import csr_matrix

# Data analysis and plotting
import pandas as pd
import seaborn as sns
import uproot

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


#import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import networkx
import energyflow

from analysis.architectures.dnn import DNN 

import random

from dataloader import read_file


class nsubDNN():
    
    #---------------------------------------------------------------
    def __init__(self, model_info):
        '''
        :param model_info: Dictionary of model info, containing the following keys:
                                'model_settings': dictionary of model settings
                                'n_total': total number of training+val+test examples
                                'n_train': total number of training examples
                                'n_test': total number of test examples
                                'torch_device': torch device
                                'output_dir':   output directory
                                'body_dim':     n-body phase space dimension
        '''
        
        #print('initializing ParT...')

        self.model_info = model_info
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'torch_device: {self.torch_device}')

        self.output_dir = model_info['output_dir']
        self.classification_task = model_info['classification_task']

        if self.classification_task not in ['ZvsQCD', 'qvsg']: 
            sys.exit('Invalid classification task. Choose between ZvsQCD and qvsg. For the potential extension to other tasks from JetClass, please check dataloader.py and modify the code accordingly.')
        
        self.n_total = model_info['n_total']
        self.n_train = model_info['n_train']
        self.n_test = model_info['n_test']
        self.n_val = model_info['n_val'] 
        self.test_frac = self.n_test/self.n_total
        self.val_frac = self.n_val/self.n_total

        self.batch_size = self.model_info['model_settings']['batch_size']
        self.learning_rate = self.model_info['model_settings']['learning_rate']
        self.epochs = self.model_info['model_settings']['epochs']
        self.K = self.model_info['model_settings']['K']

        self.output = defaultdict(list)
        self.N_list = []
        self.beta_list = []

        for i in range(self.K-2):
            self.N_list += [i+1] * 3
            self.beta_list += [0.5,1,2]

        #self.body_dim = model_info['body_dim']
        self.X_nsub, self.Y  = self.init_data()
        print(f'X_nsub.shape: {self.X_nsub.shape}')
        print(f'Y.shape: {self.Y.shape}')

        self.model = self.init_model()


    #---------------------------------------------------------------
    def init_data(self):
        '''
        Initialize data loaders
        '''
        print('initializing data loaders...')
        
        #if self.classification_task == 'qvsg':
        # Load the four-vectors directly from the quark vs gluon data set
        X_ParT, Y = energyflow.datasets.qg_jets.load(num_data=self.n_total, pad=True, 
                                                                generator='pythia',  # Herwig is also available
                                                                with_bc=False        # Turn on to enable heavy quarks
                                                            )                        # X_PFN.shape = (n_jets, n_particles per jet, n_variables)  
        
        #elif self.classification_task == 'ZvsQCD':
        #    # TODO: Load the four-vectors from the Z vs QCD data set
        #    pass 

        print(f'X_ParT.shape: {X_ParT.shape}')

        columns = ['pt', 'y', 'phi', 'pid']
        df_particles = pd.DataFrame(X_ParT.reshape(-1, 4), columns=columns)
        df_particles.index = np.repeat(np.arange(X_ParT.shape[0]), X_ParT.shape[1]) + 1
        df_particles.index.name = 'jet_id'
                
        # (i) Group the particle dataframe by jet id
        #     df_particles_grouped is a DataFrameGroupBy object with one particle dataframe per jet
        df_fjparticles_grouped = df_particles.groupby('jet_id')


        # (ii) Transform the DataFrameGroupBy object to a SeriesGroupBy of fastjet::PseudoJets
        # NOTE: for now we neglect the mass -- and assume y=eta
        # TO DO: Add y to https://github.com/matplo/heppy/blob/master/cpptools/src/fjext/fjtools.cxx
        # TO DO: Add mass vector using pdg
        print('Converting particle dataframe to fastjet::PseudoJets...')
        self.df_fjparticles = df_fjparticles_grouped.apply(self.get_fjparticles)
        print('Done.')
        print()

        
        # Fill each of the jet_variables into a list
        fj.ClusterSequence.print_banner()
        print('Finding jets and computing N-subjettiness and subjets...')
        result = [self.analyze_event(fj_particles) for fj_particles in self.df_fjparticles]

        # make self.output into a numpy array 
        self.output_np = {key: np.array(value) for key, value in self.output.items()}

        print(self.output_np.keys())

        # create a self.X_nsub that is a numpy array of shape (n_jets, n_features) where the n_features are the Nsubjettiness features
        X_nsub = np.column_stack([self.output_np[key] for key in self.output_np.keys()])
        
        return X_nsub, Y
        

    #---------------------------------------------------------------
    def init_model(self, hidden_size = 256, dropout = 0.05):
        '''
        :return: pytorch architecture
        '''

        # Define the model 
        model = DNN(input_size=self.X_nsub.shape[1], hidden_size=hidden_size, dropout = dropout)

        model = model.to(self.torch_device)

        # Print the model architecture
        print()
        print(model)
        print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
        print()
        return model
    
    #---------------------------------------------------------------
    def shuffle_and_split(self, X, Y, test_ratio=0.1, val_ratio=0.1):
        """
        Shuffles and splits the data into training, validation, and test sets.

        Parameters:
        - X: Features.
        - Y: Targets/Labels.
        - train_ratio: Proportion of the dataset to include in the train split.
        - test_ratio: Proportion of the dataset to include in the test split.
        - val_ratio: Proportion of the dataset to include in the validation split.
        - random_seed: The seed used by the random number generator.

        Returns:
        - X_train, Y_train: Training set.
        - X_val, Y_val: Validation set.
        - X_test, Y_test: Test set.
        """

        # First, split into training and temp (test + validation) sets
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=(test_ratio + val_ratio))

        # Now split the temp set into actual test and validation sets
        test_size_proportion = test_ratio / (test_ratio + val_ratio)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=test_size_proportion)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test


    #---------------------------------------------------------------
    # Train DNN, using hyperparameter optimization with keras tuner
    #---------------------------------------------------------------
    def train(self):
        print()
        print(f'Training nsub transformer...') 

        # shuffle X_nsub and self.Y and split into training/test/validation sets
        X_train, Y_train, X_val, Y_val, X_test, Y_test = self.shuffle_and_split(self.X_nsub, self.Y, test_ratio=self.test_frac, val_ratio=self.val_frac)     

        # split the training data into batches of size 64

        train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float().view(-1, 1))
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True) 

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_auc_test = 0
        best_auc_val, best_roc_val = None, None 

        time_start = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            for X_batch, Y_batch in train_loader:
                # move the data to the device            
                X_batch, Y_batch = X_batch.to(self.torch_device), Y_batch.to(self.torch_device)

                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, Y_batch)
                loss.backward()
                optimizer.step()

            # calculate the AUC 
            self.model.eval()
            with torch.no_grad():
                # Move data to the device for evaluation
                X_train_tensor = torch.tensor(X_train).float().to(self.torch_device)
                X_val_tensor = torch.tensor(X_val).float().to(self.torch_device)
                X_test_tensor = torch.tensor(X_test).float().to(self.torch_device)
                
                # Compute outputs
                output_train = self.model(X_train_tensor)
                output_val = self.model(X_val_tensor)
                output_test = self.model(X_test_tensor)
                
                # Compute AUC (move tensors to CPU for sklearn compatibility)
                auc_train = roc_auc_score(Y_train, output_train.cpu().numpy())
                auc_val = roc_auc_score(Y_val, output_val.cpu().numpy())
                roc_val = roc_curve(Y_val, output_val.cpu().numpy())
                auc_test = roc_auc_score(Y_test, output_test.cpu().numpy())

                if auc_test > best_auc_test:
                    best_auc_test = auc_test
                    best_auc_val = auc_val
                    best_roc_val = roc_val


            # with 4 decimal places
            print(f'--------------------------------------------------------------------')
            print(f"Epoch {epoch+1}: loss = {loss.item():.4f}, AUC train = {auc_train:.4f}, AUC val = {auc_val:.4f}, AUC test = {auc_test:.4f}")
        
        time_end = time.time()
        print(f'--------------------------------------------------------------------')
        print()
        print(f"Time to train model for 1 epoch = {(time_end - time_start)/self.epochs:.1f} seconds")
        print()
        print(f"Best AUC on the test set = {best_auc_test:.4f}")
        print(f"Corresponding AUC on the validation set = {best_auc_val:.4f}")
        print()

        return best_auc_val, best_roc_val


    #---------------------------------------------------------------
    # Transform particles to fastjet::PseudoJets
    #---------------------------------------------------------------
    def get_fjparticles(self, df_particles_grouped):

        user_index_offset = 0
        return fjext.vectorize_pt_eta_phi(df_particles_grouped['pt'].values,
                                          df_particles_grouped['y'].values,
                                          df_particles_grouped['phi'].values,
                                          user_index_offset)


    #---------------------------------------------------------------
    # Process an event
    #---------------------------------------------------------------
    def analyze_event(self, fj_particles):
    
        # Check that the entries exist appropriately
        if fj_particles and type(fj_particles) != fj.vectorPJ:
            print('fj_particles type mismatch -- skipping event')
            return

        # Find jets -- one jet per "event".  We only use antikt for the Jet Clustering
        jet_def = fj.JetDefinition(fj.antikt_algorithm, fj.JetDefinition.max_allowable_R)

        cs = fj.ClusterSequence(fj_particles, jet_def)
        jet_selected = fj.sorted_by_pt(cs.inclusive_jets())[0]
        #print(f'jet_selected: {jet_selected}')
        #print()
        
        # Compute jet quantities and store in our data structures
        self.analyze_jets(jet_selected)
        
        #for key in self.output.keys():
        #    print(f'{key}: {self.output[key]}')
        #    print(f'{key}.shape: {np.array(self.output[key]).shape}')
        #    print()

    #---------------------------------------------------------------
    # Analyze jets of a given event.
    #---------------------------------------------------------------
    def analyze_jets(self, jet_selected):
        self.fill_nsubjettiness(jet_selected)


    #---------------------------------------------------------------
    # Compute Nsubjettiness of jet
    #---------------------------------------------------------------
    def fill_nsubjettiness(self, jet):
    
        axis_definition = fjcontrib.OnePass_KT_Axes()

        for i,N in enumerate(self.N_list):
            
            beta = self.beta_list[i]
            measure_definition = fjcontrib.UnnormalizedMeasure(beta)
            n_subjettiness_calculator = fjcontrib.Nsubjettiness(N, axis_definition, measure_definition)
            n_subjettiness = n_subjettiness_calculator.result(jet)/jet.pt()
            
            self.output[f'N{N}_beta{beta}'].append(n_subjettiness)

