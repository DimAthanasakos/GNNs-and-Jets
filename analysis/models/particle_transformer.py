''' ParticleNet

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570

Adapted from the DGCNN implementation in https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.

Here we use 3 different versions of ParticleNet, the original one (7dim features for each particle and nearest neighbors algorithm at each layer), 
a 3dim modified one (3dim features for each particle and nearest neighbors algorithm at each layer, to make comparisons with Laman Graphs easier) 
and  a Laman one (3dim features for each particle and Laman Graphs for the first layer only, after that we use nearest neighbors).
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics

import torch
import networkx
import energyflow
from analysis.architectures import ParticleTransformer  


class ParT():
    
    #---------------------------------------------------------------
    def __init__(self, model_info):
        '''
        :param model_info: Dictionary of model info, containing the following keys:
                                'model_settings': dictionary of model settings
                                'n_total': total number of training+val+test examples
                                'n_train': total number of training examples
                                'n_test': total number of test examples
                                'torch_device': torch device
                                'output_dir': output directory
                                'Laman': boolean variable to choose between the original ParticleNet and the Laman implementation 
                                'three_momentum_features': ONLY RELEVENAT IF Laman = false. Boolean variable to choose between the 7 dimensional representation and the 3 dimensional one 

                           In the case of subjet GNNs, the following keys are also required, originating from the graph_constructor:
                                'r': subjet radius
                                'n_subjets_total': total number of subjets per jet
                                'subjet_graphs_dict': dictionary of subjet graphs
        '''
        self.model_info = model_info
        
    
        self.torch_device = model_info['torch_device']
        self.output_dir = model_info['output_dir']
        self.n_total = model_info['n_total']
        self.n_train = model_info['n_train']
        self.n_test = model_info['n_test']
        self.n_val = model_info['n_val']
        

        self.train_loader, self.val_loader, self.test_loader = self.init_data()
        self.model = self.init_model()

    #---------------------------------------------------------------
    def init_data(self):

        # Note: Currently we are only supporting the quark vs gluon dataset from the energyflow package. We can easily modify
        # the code to support our own Z vs qcd dataset as well.
        
        # Load the four-vectors directly from the quark vs gluon data set
        self.X_ParT, self.Y_ParT = energyflow.datasets.qg_jets.load(num_data=self.n_total, pad=True, 
                                                        generator='pythia',  # Herwig is also available
                                                        with_bc=False        # Turn on to enable heavy quarks
                                                       )                     # X_PFN.shape = (n_jets, n_particles per jet, n_variables)  
           

        # Preprocess by centering jets and normalizing pts
        for x_ParT in self.X_ParT:
            mask = x_ParT[:,0] > 0
            yphi_avg = np.average(x_ParT[mask,1:3], weights=x_ParT[mask,0], axis=0)
            x_ParT[mask,1:3] -= yphi_avg
            x_ParT[mask,0] /= x_ParT[:,0].sum()

        # TODO:
        # Change the architecture.ParticleTransformer script to accept (pt, eta, phi) as input features instead of (px, py, pz, E) in order to save compute time
        
        # Change the order of the features from (pt, eta, phi, pid) to (px, py, pz, E) to agree with the architecture.ParticleTransformer script
        self.X_ParT = energyflow.p4s_from_ptyphipids(self.X_ParT, error_on_unknown = True)
        # (E, px, py, pz) -> (px, py, pz, E)
        self.X_ParT[:,:, [0, 1, 2, 3]] = self.X_ParT[:,:, [1, 2, 3, 0]] 
        
        # Transpose the data to match the ParticleNet architecture convention which is (batch_size, n_features, n_particles) 
        # instead of the current (batch_size, n_particles, n_features)
        self.X_ParT = np.transpose(self.X_ParT, (0, 2, 1))

        # Split data into train, val and test sets
        (features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test) = energyflow.utils.data_split(self.X_ParT, self.Y_ParT,
                                                                                                               val=self.n_val, test=self.n_test)

        # Data loader   

        batch_size = self.model_info['model_settings']['batch_size']
        
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(Y_ParT_train).long())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_val).float(), torch.from_numpy(Y_ParT_val).long())
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(Y_ParT_test).long())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    #---------------------------------------------------------------
    def init_model(self):
        '''
        :return: pytorch architecture
        '''

        # Define the model 
        model = ParticleTransformer.ParticleTransformer(input_dim = 4, num_classes = 2) # 4 features: (px, py, pz, E)

        model = model.to(self.torch_device)
        
        return model 


    #---------------------------------------------------------------
    def train(self):
        print(f'Training...')
        print()

        time_start = time.time()
        # Use custon training parameters
        epochs = self.model_info['model_settings']['epochs']
        learning_rate = self.model_info['model_settings']['learning_rate']

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

        for epoch in range(1, epochs+1):
            print("--------------------------------")
            self._train_part(self.train_loader, self.model, optimizer, criterion)

            auc_test, acc_test, roc_test = self._test_part(self.test_loader, self.model)
            auc_val, acc_val, roc_val = self._test_part(self.val_loader, self.model)
                
            if (epoch)%5 == 0:
                auc_train, acc_train, roc_train = self._test_part(self.train_loader, self.model)
                print(f'Epoch: {epoch:02d}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')
            else:
                print(f'Epoch: {epoch:02d}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')

        time_end = time.time()
        print("--------------------------------")
        print()
        print(f"Time to train model for 1 epoch = {(time_end - time_start)/epochs} seconds")
        print()
        print()
        
        return auc_test, roc_test

        
    #---------------------------------------------------------------
    def _train_part(self, train_loader, model, optimizer, criterion):

        model.train() # Set model to training mode. This is necessary for dropout, batchnorm etc layers 
                                  # that behave differently in training mode vs eval mode (which is the default)
                                  # We need to include this since we have particlenet.eval() in the test_particlenet function
        
        loss_cum = 0              # Cumulative loss
        
        for index, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(self.torch_device)
            labels = labels.to(self.torch_device) 
            
            # we need to turn labels to one-hot encoding depending on the application (?)
            #labels_onehot = torch.nn.functional.one_hot(labels, num_classes=2).to(self.torch_device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x = inputs, v = inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_cum += loss.item()
            # Cache management
            torch.cuda.empty_cache()
        
        return loss_cum


    #---------------------------------------------------------------
    @torch.no_grad()
    def _test_part(self, test_loader, model):
        model.eval()

        correct = 0
        tot_datapoints = 0

        all_labels = []
        all_output_softmax = []

        for index, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(self.torch_device)
            labels = labels.to(self.torch_device)
            tot_datapoints += len(labels)

            outputs = model(x = inputs, v = inputs)

            output_softmax = torch.nn.functional.softmax(outputs, dim=1) # Keep on GPU
            all_output_softmax.append(output_softmax[:, 1].detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            pred = outputs.argmax(dim=1)  # No need for keepdim=True
            correct += pred.eq(labels).sum().item()

        # Calculate ROC, AUC outside the loop
        all_labels = np.concatenate(all_labels)
        all_output_softmax = np.concatenate(all_output_softmax)

        auc = sklearn.metrics.roc_auc_score(all_labels, all_output_softmax)
        roc = sklearn.metrics.roc_curve(all_labels, all_output_softmax)

        return (auc, correct / tot_datapoints, roc)

