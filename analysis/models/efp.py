'''
DNN based on the nsubjettiness features. This script is loading the already calculated nsubs from the file nsubs.h5 and training a DNN on them.

'''
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
import h5py


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


class efpDNN():
    
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
        

        self.model_info = model_info
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'torch_device: {self.torch_device}')

        self.output_dir = model_info['output_dir']

        self.n_total = model_info['n_total']
        self.n_train = model_info['n_train']
        self.n_test = model_info['n_test']
        self.n_val = model_info['n_val'] 
        self.test_frac = self.n_test/self.n_total
        self.val_frac = self.n_val/self.n_total

        self.batch_size = self.model_info['model_settings']['batch_size']
        self.learning_rate = self.model_info['model_settings']['learning_rate']
        self.epochs = self.model_info['model_settings']['epochs']
        self.dmax = self.model_info['model_settings']['d']
        self.efp_measure = self.model_info['model_settings']['efp_measure']
        self.efp_beta = self.model_info['model_settings']['efp_beta']

        self.output = defaultdict(list)
        

        self.classification_task = model_info['classification_task'] 

        self.init_data(classification_task=self.classification_task, n_total=self.n_total)



    #---------------------------------------------------------------
    def init_data(self, classification_task, n_total):
        '''
        Load the data, preprocess it by calculating the LOT coordinates, and return the input data and labels
        '''
        # Load labels and data, four vectors. Format: (pT,y,phi,m=0). Note: no PID yet which would be 5th entry... check later!

        if classification_task == 'qvsg': 
            # Load the four-vectors directly from the quark vs gluon data set
            X_EFP, Y_EFP = energyflow.datasets.qg_jets.load(num_data=n_total, pad=True, 
                                                                        generator='pythia',  # Herwig is also available
                                                                        with_bc=False        # Turn on to enable heavy quarks
                                                                        )                    # X_PFN.shape = (n_jets, n_particles per jet, n_variables)  
        else: 
            # Each file contains 100k jets for each class
            n_signal = n_bckg = n_total // 2 
            #/pscratch/sd/d/dimathan/JetClass_Dataset/t_jets/bqq
            directory_path = '/pscratch/sd/d/dimathan/JetClass_Dataset'
            if classification_task == 'ZvsQCD':   
                signal_jet_filepattern=  f"{directory_path}/Z_jets/ZToQQ*"
                label_signal = 'label_Zqq'
            elif classification_task == 'TvsQCD': 
                signal_jet_filepattern=  f"{directory_path}/t_jets/bqq/TTBar*"
                label_signal = 'label_Tbqq'

            bckg_jet_filepattern = f"{directory_path}/qg_jets/ZJetsToNuNu*"

            # Getting the list of files that match the patterns
            signal_jet_files = glob.glob(signal_jet_filepattern)
            bckg_jet_files = glob.glob(bckg_jet_filepattern)
            
            x_particles_signal, x_jet_signal, y_signal = np.array([]), np.array([]), np.array([]) 
            for file in signal_jet_files:
                x_particles, x_jet, y = read_file(filepath = file, labels = [label_signal, 'label_QCD'])
                x_particles_signal = np.concatenate((x_particles_signal, x_particles), axis = 0) if x_particles_signal.size else x_particles
                x_jet_signal = np.concatenate((x_jet_signal, x_jet), axis = 0) if x_jet_signal.size else x_jet
                y_signal = np.concatenate((y_signal, y), axis = 0) if y_signal.size else y
                if x_particles_signal.shape[0] >= n_signal: 
                    x_particles_signal = x_particles_signal[:n_signal]
                    x_jet_signal = x_jet_signal[:n_signal]
                    y_signal = y_signal[:n_signal]
                    break # Stop reading files if we have enough jets
                
            x_particles_bckg, x_jet_bckg, y_bckg = np.array([]), np.array([]), np.array([])
            for file in bckg_jet_files:
                x_particles, x_jet, y = read_file(filepath = file, labels = [label_signal, 'label_QCD'])
                x_particles_bckg = np.concatenate((x_particles_bckg, x_particles), axis = 0) if x_particles_bckg.size else x_particles
                x_jet_bckg = np.concatenate((x_jet_bckg, x_jet), axis = 0) if x_jet_bckg.size else x_jet
                y_bckg = np.concatenate((y_bckg, y), axis = 0) if y_bckg.size else y
                if x_particles_bckg.shape[0] >= n_bckg: 
                    x_particles_bckg = x_particles_bckg[:n_bckg]
                    x_jet_bckg = x_jet_bckg[:n_bckg]
                    y_bckg = y_bckg[:n_bckg]
                    break
            
            # concatenate the two datasets 
            X_ParT = np.concatenate((x_particles_signal, x_particles_bckg), axis = 0)
            Y_ParT = np.concatenate((y_signal, y_bckg), axis = 0)
            x_jet = np.concatenate((x_jet_signal, x_jet_bckg), axis = 0)

            print()
            print(f"Found {len(signal_jet_files)} files matching {classification_task} pattern.")
            print(f"Found {len(bckg_jet_files)} files matching ZJetsToNuNu pattern.")
            print()
            print(f"Loaded {X_ParT.shape[0]} jets for the {classification_task} classification task.")
            print()
            

            # match the shape of the data to the shape of the energyflow data for consistency
            Y_EFP = Y_ParT[:, 0] # one-hot encoding, where 0: Background (QCD) and 1: Signal (Z) 
            X_EFP = np.transpose(X_ParT, (0, 2, 1))

        print()
        print(f'Calculating d <= {self.dmax} EFPs for {self.n_total} jets... ')
        t_start = time.time()

        efpset = energyflow.EFPSet(('d<=', self.dmax), measure=self.efp_measure, beta=self.efp_beta)

        # Convert to list of np.arrays of jets in format (pT,y,phi,mass or PID) -> dim: (# jets, # particles in jets, #4)
        # and remove zero entries
        masked_X_EFP = [x[x[:,0] > 0] for x in X_EFP]
                            
        # Now compute EFPs
        X_EFP = efpset.batch_compute(masked_X_EFP)


        # Record which EFPs correspond to which indices
        # Note: graph images are available here: https://github.com/pkomiske/EnergyFlow/tree/images/graphs
        self.graphs = efpset.graphs()[1:]
        for i,efp in enumerate(self.graphs):
            print(f'  efp {i} -- edges: {efp}')

        # Preprocess, plot, and store the EFPs for each d
        self.X_EFP_train = {}
        self.X_EFP_test = {}
        self.Y_EFP_train = {}
        self.Y_EFP_test = {}
                                    
        for d in range(1, self.dmax+1):

            # Select EFPs with degree <= d
            X_EFP_d = X_EFP[:,efpset.sel(('d<=', d))]

            # Remove the 0th EFP (=1)
            X_EFP_d = X_EFP_d[:,1:]
            print(f'There are {X_EFP_d.shape[1]} terms for d<={d} (connected + disconnected, and excluding d=0)')

            # Plot EFPs
            #if d == 2:
            #    self.plot_efp_distributions(d, X_EFP_d, suffix='before_scaling')
            #    self.plot_efp_distributions(d, sklearn.preprocessing.scale(X_EFP_d.astype(np.float128)), suffix='after_scaling')

            # Do train/val/test split (Note: separate val_set generated in DNN training.)
            (X_EFP_train_d, X_EFP_val, 
            self.X_EFP_test[d], self.Y_EFP_train[d], 
            Y_EFP_val, self.Y_EFP_test[d]) = energyflow.utils.data_split(X_EFP_d, Y_EFP, val=self.n_val, test=self.n_test)
                                
            # Preprocessing: zero mean unit variance
            self.X_EFP_train[d] = sklearn.preprocessing.scale(X_EFP_train_d.astype(np.float128))
        print(f'It took {time.time() - t_start:.1f} seconds to calculate the EFPs.')
        print()

        print('data loaded')
        print()

        return self.X_EFP_train, self.Y_EFP_train, self.X_EFP_test, self.Y_EFP_test


    #---------------------------------------------------------------
    def init_model(self, X_train, hidden_size = 256, dropout = 0.1):
        '''
        :return: pytorch architecture
        '''

        # Define the model 
        model = DNN(input_size=X_train.shape[1], hidden_size=hidden_size, dropout = dropout)

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
        for d in range(1, self.dmax+1):
            print(f'Training efp DNN with d={d}...') 

            X_train = self.X_EFP_train[d]
            X_test = self.X_EFP_test[d]
            Y_train = self.Y_EFP_train[d]
            Y_test = self.Y_EFP_test[d]


            self.model = self.init_model(X_train)


            # split the training data into batches of size 64
            train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float().view(-1, 1))
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True) 

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            best_auc_test = 0
            #best_auc_val, best_roc_val = None, None 
            best_roc_test = None

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
                    #X_val_tensor = torch.tensor(X_val).float().to(self.torch_device)
                    X_test_tensor = torch.tensor(X_test).float().to(self.torch_device)
                    
                    # Compute outputs
                    output_train = self.model(X_train_tensor)
                    #output_val = self.model(X_val_tensor) 
                    output_test = self.model(X_test_tensor)
                    
                    # Compute AUC (move tensors to CPU for sklearn compatibility)
                    auc_train = roc_auc_score(Y_train, output_train.cpu().numpy())
                    #auc_val = roc_auc_score(Y_val, output_val.cpu().numpy())
                    #roc_val = roc_curve(Y_val, output_val.cpu().numpy())
                    auc_test = roc_auc_score(Y_test, output_test.cpu().numpy())
                    roc_test = roc_curve(Y_test, output_test.cpu().numpy())

                    if auc_test > best_auc_test:
                        best_auc_test = auc_test
                        #best_auc_val = auc_val
                        #best_roc_val = roc_val
                        best_roc_test = roc_test

                    # lets calculate accurace in val and test: 
                    #val_accuracy = sklearn.metrics.accuracy_score(Y_val, output_val.cpu().numpy()> 0.) # >0 and not >0.5 because we are not using sigmoid activation
                    test_accuracy = sklearn.metrics.accuracy_score(Y_test, output_test.cpu().numpy() > 0.)

                    # print the first 10 predictions and true values
                    #print(f'--------------------------------------------------------------------')
                    #print(f'self.model(X_val_tensor).shape: {self.model(X_val_tensor).shape}')
                    #print(f'Predictions: {(output_val.cpu().numpy())[:10]}')
                    #print(f'True values: {Y_val[:10]}')

                # with 4 decimal places
                print(f'--------------------------------------------------------------------')
                print(f"Epoch {epoch+1}: loss = {loss.item():.4f}, AUC train = {auc_train:.4f}, test accuracy = {test_accuracy:.4f}, AUC test = {auc_test:.4f}")
            
            time_end = time.time()
            print(f'--------------------------------------------------------------------')
            print()
            print(f"Time to train model for 1 epoch = {(time_end - time_start)/self.epochs:.1f} seconds")
            print(f'd: {d}')
            print(f"Best AUC on the test set = {best_auc_test:.4f}")
            print(f"Corresponding AUC on the test set = {best_auc_test:.4f}")
            print()

        return best_auc_test, best_roc_test




#---------------------------------------------------------------
if __name__ == '__main__':
    model_info = {
        'output_dir': '/pscratch/sd/d/dimathan/OT/test_output',
        'n_total': 2000,
        'n_train': 1600, 
        'n_test': 200,
        'n_val': 200,
        'model_settings': {
            'epochs':10,
            'learning_rate':0.0002,
            'batch_size':256,
            'd': 5,
        }
    }
    classifier = efpDNN(model_info)
    classifier.train()