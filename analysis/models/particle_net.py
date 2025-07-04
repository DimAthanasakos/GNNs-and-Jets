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
import sys

import glob

import torch
import networkx
import energyflow
from analysis.architectures import ParticleNet_Laman 

from dataloader import read_file

class ParticleNet():
    
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
        
        if model_info['model_key'] == 'particle_net_laman': self.Laman = True
        else: self.Laman = False

        if 'three_momentum_features' in model_info['model_settings']: self.three_momentum_features = model_info['model_settings']['three_momentum_features']
        else: self.three_momentum_features = False

        # The original ParticleNet uses a niche choice of hyperparameters for training. Check the train function for more details.
        # So far it offers the best performance for the quark vs gluon dataset from the energyflow package out of all the models we have tested.
        if 'original_train_parameters' in model_info['model_settings']: self.original_train_parameters = model_info['model_settings']['original_train_parameters']
        else: self.original_train_parameters = False
        
        self.torch_device = model_info['torch_device']
        self.output_dir = model_info['output_dir']

        self.classification_task = model_info['classification_task']
        
        self.n_total = model_info['n_total']
        self.n_train = model_info['n_train']
        self.n_test = model_info['n_test']
        self.n_val = model_info['n_val']
        

        self.train_loader, self.val_loader, self.test_loader = self.init_data(classification_task=self.classification_task, n_total=self.n_total)
        self.model = self.init_model()

    #---------------------------------------------------------------
    def init_data(self, classification_task, n_total):
        print()
        if self.Laman: print(f"Training particle_net w/ Laman graphs ...")
        else: print(f"Training with original particle_net ...")

        if self.Laman and not self.three_momentum_features:
            print(f"WARNING: You are using Laman Graphs with a 7 dimensional representation.") 
            print(f"This is not currently supported. Overriding the three_momentum_features variable to True.")
            self.three_momentum_features = True
        
        if classification_task == 'qvsg': 
            # Load the four-vectors directly from the quark vs gluon data set
            self.X_PN, self.Y_PN = energyflow.datasets.qg_jets.load(num_data=n_total, pad=True, 
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
            X_PN = np.concatenate((x_particles_signal, x_particles_bckg), axis = 0)
            Y_PN = np.concatenate((y_signal, y_bckg), axis = 0)
            x_jet = np.concatenate((x_jet_signal, x_jet_bckg), axis = 0)

            print()
            print(f"Found {len(signal_jet_files)} files matching {classification_task} pattern.")
            print(f"Found {len(bckg_jet_files)} files matching ZJetsToNuNu pattern.")
            print()
            print(f"Loaded {X_PN.shape[0]} jets for the {classification_task} classification task.")
            print()
            

            # match the shape of the data to the shape of the energyflow data for consistency
            self.Y_PN = Y_PN[:, 0] # one-hot encoding, where 0: Background (QCD) and 1: Signal (Z) 
            self.X_PN = np.transpose(X_PN, (0, 2, 1))



        # Preprocess by centering jets and normalizing pts
        if self.three_momentum_features:                 # Preprocess the jets to create the three_momentum_features for ParticleNet 
            self.X_PN = self.X_PN[:,:,:3]
            input_dims = 3
            for x_PN in self.X_PN:
                mask = x_PN[:,0] > 0
                yphi_avg = np.average(x_PN[mask,1:3], weights=x_PN[mask,0], axis=0)
                x_PN[mask,1:3] -= yphi_avg
                x_PN[mask,0] /= x_PN[:,0].sum()

            # Change the order of the features from (pt, eta, phi, pid) to (eta, phi, pt, pid) because (eta, phi) are common for both representations
            # So the convention for the order is: (eta, phi, ...) 
            self.X_PN[:,:, [0, 1, 2]] = self.X_PN[:,:, [1, 2, 0]]

            # Ignore the PID features and use the array features = (eta, phi, pt) as input features for ParticleNet
            features = self.X_PN[:,:,:3]  

        # Preprocess the jets to create the original features for ParticleNet (7 in total)
        else:
            
            # TO DO: Check if the functions from architecture.ParticleTransformer can be used here to make this more readable

            # Get the four-momentum (E, px, py, pz) from the (pt, eta, phi, pid) features
            p_4momentum = energyflow.p4s_from_ptyphipids(self.X_PN, error_on_unknown = True)

            # Transform pids to masses 
            self.X_PN[:,:,3] = energyflow.pids2ms(self.X_PN[:,:,3], error_on_unknown = True)

            # Use the array features = (eta, phi, logpt, logE, logpt/pt_jet, logE/E_jet, ΔR) as input features for ParticleNet
            input_dims = 7
            features_shape = (self.X_PN.shape[0], self.X_PN.shape[1], input_dims) 
            features = np.zeros(features_shape)
            
            
            for index_jet, jet in enumerate(self.X_PN):

                # Get the jet coordinates (pt, y, phi, mass) from the subjets' (pt, eta, phi, mass) 
                jet_coords = energyflow.sum_ptyphims(jet, scheme='escheme')
                pt_jet, y_jet, phi_jet, mass_jet = jet_coords[0], jet_coords[1], jet_coords[2], jet_coords[3]
                eta_jet = energyflow.etas_from_pts_ys_ms(pt_jet, y_jet, mass_jet)
                E_jet = np.cosh(y_jet)*np.sqrt(pt_jet**2 + mass_jet**2)
                

                mask = jet[:,0] > 0 # Select only the particles with pt > 0. This is needed to avoid log(0)  
                                    # since we need to calculate log(pt) and log(E) for the particles 

                # Transform the y's to eta's in accordance with the ParticleNet paper
                jet[mask, 1] = energyflow.etas_from_pts_ys_ms(jet[mask, 0], jet[mask, 1], jet[mask, 3]) 

                # Center the subjet coordinates (pt, eta, phi, mass) around the jet axis
                jet[mask, :] = energyflow.center_ptyphims(jet[mask, :], axis = [eta_jet, phi_jet], center='escheme', copy=True)

                delta_r = np.sqrt((jet[mask,1])**2 + (jet[mask,2])**2)
                
                features[index_jet, mask, 0] = jet[mask, 1]                                             # Δη    
                features[index_jet, mask, 1] = jet[mask, 2]                                             # Δφ
                features[index_jet, mask, 2] = np.log(jet[mask, 0])                                     # logpt
                features[index_jet, mask, 3] = np.log(p_4momentum[index_jet, mask, 0])                  # logE
                features[index_jet, mask, 4] = np.log(jet[mask, 0]) - np.log(pt_jet)                    # logpt/pt_jet
                features[index_jet, mask, 5] = np.log(p_4momentum[index_jet, mask, 0]) - np.log(E_jet)  # logE/E_jet
                features[index_jet, mask, 6] = delta_r                                                  # ΔR
                

        # Split data into train, val and test sets
        (features_train, features_val, features_test, Y_PN_train, Y_PN_val, Y_PN_test) = energyflow.utils.data_split(features, self.Y_PN,
                                                                                                               val=self.n_val, test=self.n_test)

        # Data loader   

        batch_size = self.model_info['model_settings']['batch_size']
        
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(Y_PN_train).long())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_val).float(), torch.from_numpy(Y_PN_val).long())
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(Y_PN_test).long())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    #---------------------------------------------------------------
    def init_model(self):
        '''
        :return: pytorch architecture
        '''

        # Define the model 
        particlenet_model = ParticleNet_Laman.ParticleNet(input_dims = 3 if self.three_momentum_features else 7, num_classes = 2, Laman = self.Laman)

        particlenet_model = particlenet_model.to(self.torch_device)
        
        print()
        print(f"particle_net model: {particlenet_model}")
        print()
        
        print(particlenet_model)
        print(f'Total number of parameters: {sum(p.numel() for p in particlenet_model.parameters())}')
        print()

        return particlenet_model 


    #---------------------------------------------------------------
    def train(self):
        print(f'Training...')
        print()


        if self.original_train_parameters:

            # Use the original training parameters from the ParticleNet paper
            # The original ParticleNet paper uses AdamW (Adam with weight decay)
            epochs = 20 
            cooldown_epochs = 4
            initial_lr = 0.0005  # Replace with 5e-4 for ParticleNet-Lite
            peak_lr = 0.005    # Replace with 5e-3 for ParticleNet-Lite
            final_lr = 0.000001   # Replace with 1e-6 for ParticleNet-Lite
            weight_decay = 0.0001
            steps_per_epoch = len(self.train_loader)
            criterion = torch.nn.CrossEntropyLoss()

            # Set up the optimizer
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=initial_lr, weight_decay=weight_decay)

            # Set up the scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=peak_lr, epochs=epochs, steps_per_epoch=steps_per_epoch, anneal_strategy='linear', div_factor=peak_lr/initial_lr)

            time_start = time.time()
            for epoch in range(epochs - cooldown_epochs):
                print("--------------------------------")
                try:
                    print(f"Epoch {epoch+1}/{epochs}, Current LR: {scheduler.get_last_lr()}")
                except:
                    pass 
                self._train_particlenet(self.train_loader, self.model, optimizer, criterion, scheduler=scheduler)


                auc_test, acc_test, roc_test = self._test_particlenet(self.test_loader, self.model)
                auc_val, acc_val, roc_val = self._test_particlenet(self.val_loader, self.model)
                
                if (epoch+1)%5 == 0:
                    auc_train, acc_train, roc_train = self._test_particlenet(self.train_loader, self.model)
                    print(f'Epoch: {epoch+1:02d}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')
                else:
                    print(f'Epoch: {epoch+1:02d}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')

            
            cooldown_learning_rate = 0.0001
            cooldown_optimizer = torch.optim.Adam(self.model.parameters(), lr = cooldown_learning_rate)
            #cooldown_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=scheduler.get_last_lr()[0]/initial_lr, end_factor=final_lr/initial_lr, total_iters=cooldown_epochs*steps_per_epoch)

            for epoch in range(epochs - cooldown_epochs, epochs ):
                print("--------------------------------")
                print(f"Epoch {epoch+1}/{epochs}, Current LR: {cooldown_learning_rate}")

                for index, data in enumerate(self.train_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.torch_device)
                    labels = labels.to(self.torch_device) 
                    
                    points = inputs[:, :, :2]   # the eta-phi points of the particles to use as the points for the k-NN algorithm

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(points = points, features = inputs, mask = None)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    cooldown_optimizer.step()

                    # Cache management
                    torch.cuda.empty_cache()

                auc_test, acc_test, roc_test = self._test_particlenet(self.test_loader, self.model)
                auc_val, acc_val, roc_val = self._test_particlenet(self.val_loader, self.model)

                if (epoch+1)%5 == 0:
                    auc_train, acc_train, roc_train = self._test_particlenet(self.train_loader, self.model)
                    print(f'Epoch: {epoch+1:02d}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')
                else:
                    print(f'Epoch: {epoch+1:02d}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')
                    

            print("--------------------------------")   
            time_end = time.time()
            print()
            print(f"Time to train model for 1 epoch = {(time_end - time_start)/epochs} seconds")
            print()
            print()
       
        elif not self.original_train_parameters:

            time_start = time.time()
            # Use custon training parameters
            epochs = self.model_info['model_settings']['epochs']
            learning_rate = self.model_info['model_settings']['learning_rate']

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

            for epoch in range(1, epochs+1):
                print("--------------------------------")
                self._train_particlenet(self.train_loader, self.model, optimizer, criterion)

                auc_test, acc_test, roc_test = self._test_particlenet(self.test_loader, self.model)
                auc_val, acc_val, roc_val = self._test_particlenet(self.val_loader, self.model)
                
                if (epoch)%5 == 0:
                    auc_train, acc_train, roc_train = self._test_particlenet(self.train_loader, self.model)
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
    def _train_particlenet(self, train_loader, particlenet_model, optimizer, criterion, scheduler = None):

        particlenet_model.train() # Set model to training mode. This is necessary for dropout, batchnorm etc layers 
                                  # that behave differently in training mode vs eval mode (which is the default)
                                  # We need to include this since we have particlenet.eval() in the test_particlenet function
        
        loss_cum = 0              # Cumulative loss
        
        for index, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(self.torch_device)
            labels = labels.to(self.torch_device) 
            
            # we need to turn labels to one-hot encoding depending on the application (?)
            #labels_onehot = torch.nn.functional.one_hot(labels, num_classes=2).to(self.torch_device)
            
            points = inputs[:, :, :2]   # the eta-phi points of the particles to use as the points for the k-NN algorithm

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = particlenet_model(points = points, features = inputs, mask = None)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # this will run only for the original_train parameters where we have a variable learning rate
            try: scheduler.step()
            except: pass
 
            loss_cum += loss.item()
            # Cache management
            torch.cuda.empty_cache()
        
        return loss_cum


    #---------------------------------------------------------------
    @torch.no_grad()
    def _test_particlenet(self, test_loader, particlenet_model):
        particlenet_model.eval()

        correct = 0
        tot_datapoints = 0

        all_labels = []
        all_output_softmax = []

        # calculate the inference time in ms
        #time_start = time.time()
        for index, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(self.torch_device)
            labels = labels.to(self.torch_device)
            tot_datapoints += len(labels)

            points = inputs[:, :, :2]
            outputs = particlenet_model(points=points, features=inputs, mask=None)

            output_softmax = torch.nn.functional.softmax(outputs, dim=1) # Keep on GPU
            all_output_softmax.append(output_softmax[:, 1].detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            pred = outputs.argmax(dim=1)  # No need for keepdim=True
            correct += pred.eq(labels).sum().item()
        
        #time_end = time.time()
        #print(f"Time to test model for  = {(time_end - time_start):.6f} seconds")

        # Calculate ROC, AUC outside the loop
        all_labels = np.concatenate(all_labels)
        all_output_softmax = np.concatenate(all_output_softmax)

        auc = sklearn.metrics.roc_auc_score(all_labels, all_output_softmax)
        roc = sklearn.metrics.roc_curve(all_labels, all_output_softmax)

        return (auc, correct / tot_datapoints, roc)

