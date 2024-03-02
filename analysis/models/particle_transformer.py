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
import math 

import matplotlib.pyplot as plt
import sklearn

import torch
import networkx
import energyflow
from analysis.architectures import ParticleTransformer  
import random


def laman_graph(x): # For now this is not used 
    batch_size, _, seq_len = x.shape
    indices = np.zeros((batch_size, seq_len, seq_len))
    for i in range(seq_len-2):
        indices[:, i, i+1] = 1
        indices[:, i, i+2] = 1
    indices[seq_len-2, seq_len-1] = 1 
    
    return indices

def angles_laman(x, mask, angles = 0, pairwise_distance = None):

    batch_size, _, num_particles = mask.size()
    non_zero_particles = np.linalg.norm(x, axis=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)
    
    # remove angles
    if angles < 0:
        angles = abs(angles)
        for b in range(batch_size):
            # Generate a random permutation of n-2 numbers that starts with 2 
            if valid_n[b] <= 2:
                continue 
            permutation = torch.randperm(valid_n[b]-2) + 2      # Random generator with replacement excluding the 2 hardest particles 
                
            for i in range(min(angles, valid_n[b]-2)):          # Currently we remove angles in such a way so to ensure that the graph is still connected
                                                                # This creates a problem when we have a small number of particles. 
                                                                # My guess is that for add_angles <~ 15 it doesn't matter on average. 
                                                                # TODO: Check this with a more systematic study.
                                                                # An improved way, but slower computationally, to ensure full connectivity is:
                                                                # after transposing the adj, we remove one connection at random for particle "index" 
                                                                # if we ensure that there are other edges connecting this particle to the graph 
                                                                # NOTE: This could lead to two disconnected graphs potentially. 
                                                                # since n_edges >= n-1 for a connected graph, add_angles can at most be of order n anyways
                                                                # and the first way ensures that we have a single connected graph
                index = permutation[i]

                # the first True in the i-th row of the bool mask turned to False. SYSTEMATIC ERROR: We always remove connectivity to the hardest particle 
                first_true_index = np.argmax(mask[b, index]) # maybe alternate betwen first and second True. ?? 
                mask[b, index, first_true_index] = False
    
    # Add angles: For a particle i, add an edge to a particle j, where j < i, if there is no edge between i and j.
    elif angles > 0:
        # Mask the positions of the current edges so to make sorting easier 
        pairwise_distance[mask] = -float('inf')
        for b in range(batch_size):
            if valid_n[b] <= 3:
                continue 
            
            # mask the padded particles, i.e n >= valid_n[b]
            pairwise_distance[b, valid_n[b]:, :] = -float('inf')
            pairwise_distance[b, :, valid_n[b]:] = -float('inf')

            # Flatten the matrix
            flat_matrix = pairwise_distance[b].flatten()
            # Sort the flattened matrix
            sorted_values, flat_sorted_indices = torch.sort(flat_matrix, descending = True)
            # Convert flat indices to 2D row and column indices
            row_indices, col_indices = flat_sorted_indices//num_particles, flat_sorted_indices%num_particles

            max_angles = math.comb(valid_n[b], 2) - (2*valid_n[b] - 3) # The maximum number of angles we can add until it becomes a fully connected graph
            
            mask[b, row_indices[:min(angles, max_angles)], col_indices[:min(angles, max_angles)]] = True
                    
    return mask 

# Create a Laman Graph using a mod of the k nearest neighbors algorithm.
def knn(x, angles = 0):   
    x = torch.from_numpy(x) 

    batch_size, _, num_particles = x.size()
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)

    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    
    x = torch.cat((rapidity, phi), dim=1) # (batch_size, 2, num_points)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)                                    # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                               # (batch_size, num_points, num_points)

    # Connect the 3 hardest particles in the jet in a triangle 
    idx_3 = pairwise_distance[:, :3, :3].topk(k=3, dim=-1) # (batch_size, 3, 2)
    idx_3 = [idx_3[0][:,:,1:], idx_3[1][:,:,1:]] # (batch_size, 3, 1)
    
    # Connect the rest of the particles in a Henneberg construction: Connect the i-th hardest particle with the 2 closest particles, i_1 and i_2, where i_1,2 < j  
    pairwise_distance = pairwise_distance[:, 3:, :] # Remove the pairwise distances of 3 hardest particles from the distance matrix 
    
    # Make the upper right triangle of the distance matrix infinite so that we don't connect the i-th particle with the j-th particle if i > j 
    pairwise_distance = torch.tril(pairwise_distance, diagonal=2) - torch.triu(torch.ones_like(pairwise_distance)*float('inf'), diagonal=3)  # -inf because topk indices return the biggest values -> we've made all distances negative 


    # Find the indices of the 2 nearest neighbors for each particle
        
    idx = pairwise_distance.topk(k=2, dim=-1) # It returns two things: values, indices 
    idx = idx[1] # (batch_size, num_points - 3, 2)
        
    # Concatenate idx and idx_3 to get the indices of the 3 hardest particles and the 2 nearest neighbors for the rest of the particles
    idx = torch.cat((idx_3[1], idx), dim=1) # (batch_size, num_points, 3)
    
    # add 3 rows of -inf to the top of the pairwise_distance tensor to make it of shape (batch_size, num_particles, num_particles)
    # this is because we remove the 3 hardest particles from the graph and we don't want to connect them to the rest of the particles
    pairwise_distance = torch.cat((torch.ones((batch_size, 3, num_particles))*float('-inf'), pairwise_distance), dim=1)

    # Initialize a boolean mask with False (indicating no connection) for all pairs
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool)

    # Efficiently populate the boolean mask based on laman_indices
    for i in range(2):  # Assuming each particle is connected to two others as per laman_indices
        # Extract the current set of indices indicating connections
        current_indices = idx[:, :, i]

        # Generate a batch and source particle indices to accompany current_indices for scatter_
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_particles)
        src_particle_indices = torch.arange(num_particles).expand(batch_size, -1)

        # Use scatter_ to update the bool_mask; setting the connection locations to True
        bool_mask[batch_indices, src_particle_indices, current_indices] = True

    # remove the entries at 01, 02, 12 to be consistent with the structure of the bool mask so far is a lower triangular matrix. 
    # Those entries will be activated to True in bool_mask | bool_mask.transpose(1, 2)
    bool_mask[:, 0, 1] = False
    bool_mask[:, 0, 2] = False
    bool_mask[:, 1, 2] = False
    
    # Remove some angles at random between the particles. Default value of angles = 0.
    bool_mask = angles_laman(x, bool_mask, angles, pairwise_distance = pairwise_distance) 

    # Make the Laman Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    bool_mask = bool_mask.numpy() 

    return bool_mask 


def edge_indices_to_boolean_adjacency(adjacency_matrices, n):
    batch_size, _, max_edges = adjacency_matrices.shape
    # Initialize boolean adjacency matrix with False
    boolean_adjacency = np.zeros((batch_size, n, n), dtype=bool)
    
    for b in range(batch_size):
        for e in range(max_edges):
            i, j = adjacency_matrices[b, :, e]
            # Check for padding and update adjacency matrix
            if i != -1 and j != -1:
                boolean_adjacency[b, i, j] = True
    
    return boolean_adjacency

def rand_graph(x):
    batch_size, _, n = x.shape
    max_edges = 2 * n - 3 
    adjacency_matrices = np.full((batch_size, 2, max_edges), -1, dtype=np.int64)
    ns = []
    for b in range(batch_size):
        # Identify non-zero particles
        non_zero_particles = np.linalg.norm(x[b], axis=0) != 0

        valid_n = non_zero_particles.sum().item()
        ns.append(valid_n)
        edges = []

        num_edges = 2 * valid_n - 3
        if num_edges >= 1:    
            added_edges = set()
                
            while len(added_edges) < num_edges:
                # Select two different valid nodes
                i, j = random.sample(range(valid_n), 2)
                edge = (min(i, j), max(i, j))

                if edge not in added_edges:
                    added_edges.add(edge)
                    # Add edge indices for non-zero particles only
                    edges.append(edge) 
        else: # edge case with n=1
            edges = [(0, 0)] 
        
        # Convert edges to a sparse tensor
        # Edges and their transposes (since the graph is undirected)
        edge_indices = np.array(edges, dtype=np.int64).T
        
        try:
            adjacency_matrices[b, :, :edge_indices.shape[1]] = edge_indices
        except: 
            print(f"b: {b}, edge_indices.shape: {edge_indices.shape}, adjacency_matrices.shape: {adjacency_matrices.shape}")
            print(f"max_edges: {max_edges}, valid_n: {valid_n}, num_edges: {num_edges}")
            print(f"x[b]: {x[b]}")
            
            #time.sleep(40)
  
    adjacency_matrices = edge_indices_to_boolean_adjacency(adjacency_matrices, n)
    
    adjacency_matrices = adjacency_matrices | adjacency_matrices.transpose(1, 2)

    return adjacency_matrices


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
        self.batch_size = self.model_info['model_settings']['batch_size']
        
        self.load_model = model_info['model_settings']['load_model'] # Load a pre-trained model or not
        self.save_model = model_info['model_settings']['save_model'] # Save the model or not

        if 'random_graph' in model_info['model_settings']: self.random_graph = model_info['model_settings']['random_graph']
        else: self.random_graph = False
        
        if 'add_angles' in model_info['model_settings']: self.add_angles = model_info['model_settings']['add_angles']
        else: self.add_angles = 0
      
        self.input_dim = model_info['model_settings']['input_dim'] # 4 for (px, py, pz, E) as input for each particle, 1 for pt
        if self.input_dim not in [1, 4]:
            raise ValueError('Invalid input_dim at the config file for ParT. Must be 1 or 4') 
        
        self.pair_input_dim = model_info['model_settings']['pair_input_dim']  # how many interaction terms for pair of particles. 
                                                                              # If 3: use (dR, k_t = min(pt_1, pt_2)*dR, z = min(pt_1, pt_2)/(pt_1 + pt_2) ),
                                                                              # if 4: also use m^2 = (E1 + E2)^2 - (p1 + p2)^2    

        if model_info['model_key'].endswith('laman'):
            self.laman = True
        else: 
            self.laman = False

        #if self.laman and self.input_dim == 4: 
        #    raise ValueError('Invalid input_dim at the config file for ParT. Must be 1 for Laman Graphs')
            
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
        # Change the architecture.ParticleTransformer script to accept (pt, eta, phi) as input features for the interaction terms instead of (px, py, pz, E) in order to save compute time
        # The input terms for each particle are left as given in the ParticleTransformer architecture.
            
        # Change the order of the features from (pt, eta, phi, pid) to (px, py, pz, E) to agree with the architecture.ParticleTransformer script
        self.X_ParT = energyflow.p4s_from_ptyphipids(self.X_ParT, error_on_unknown = True)

        # (E, px, py, pz) -> (px, py, pz, E)
        self.X_ParT[:,:, [0, 1, 2, 3]] = self.X_ParT[:,:, [1, 2, 3, 0]] 
        
        # Transpose the data to match the ParticleNet architecture convention which is (batch_size, n_features, n_particles) 
        # instead of the current shape (batch_size, n_particles, n_features)
        self.X_ParT = np.transpose(self.X_ParT, (0, 2, 1))

        if self.laman:
            # we need to sort based on pt for the Laman Graphs 
            sorted_indices = np.argsort( -(self.X_ParT[:, 0, :]**2 + self.X_ParT[:, 1, :]**2), axis=-1)
            self.X_ParT = np.take_along_axis(self.X_ParT, sorted_indices[:, np.newaxis, :], axis=2)

            print(f"self.X_ParT.shape = {self.X_ParT.shape}")
            t_st = time.time()
            if self.random_graph: 
                graph = rand_graph(self.X_ParT)
            else: 
                # We need to constuct the graph in chunks to avoid memory issues when n_total > 10^6
                chunk_size = 10*1024  # Adjust this based on your memory constraints and the size of self.X_ParT
                total_size = self.X_ParT.shape[0]  # Assuming the first dimension is the batch size
                chunks = (total_size - 1) // chunk_size + 1  # Calculate how many chunks are needed

                graph = np.concatenate([knn(self.X_ParT[i * chunk_size:(i + 1) * chunk_size], angles = self.add_angles) for i in range(chunks)])

            print(f"Time to create the graph = {time.time() - t_st} seconds")

            (features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test, 
            graph_train, graph_val, graph_test) = energyflow.utils.data_split(self.X_ParT, self.Y_ParT, graph,
                                                                              val=self.n_val, test=self.n_test, shuffle = False)

            # Data loader   
        
            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(Y_ParT_train).long(), torch.from_numpy(graph_train).bool() )
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

            val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_val).float(), torch.from_numpy(Y_ParT_val).long(), torch.from_numpy(graph_val).bool() )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size, shuffle=True)
            
            test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(Y_ParT_test).long(), torch.from_numpy(graph_test).bool() ) 
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True)

        else: 
            (features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test) = energyflow.utils.data_split(self.X_ParT, self.Y_ParT,
                                                                                                                               val=self.n_val, test=self.n_test)
                         
            # Data loader   
            
            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(Y_ParT_train).long())
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

            val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_val).float(), torch.from_numpy(Y_ParT_val).long())
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size, shuffle=True)
            
            test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(Y_ParT_test).long()) 
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    #---------------------------------------------------------------
    def init_model(self):
        '''
        :return: pytorch architecture
        '''

        # Define the model 
        model = ParticleTransformer.ParticleTransformer(input_dim = self.input_dim, num_classes = 2, pair_input_dim = self.pair_input_dim) # 4 features: (px, py, pz, E)

        model = model.to(self.torch_device)
        #model = torch.compile(model)
        
        # Print the model architecture
        print()
        print(model)
        print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
        print()

        if self.load_model:
            self.path = f'/global/homes/d/dimathan/Laman-Graphs-and-Jets/{self.model_info["model_key"]}_p{self.input_dim}_{self.pair_input_dim}.pth'
            print(f"Loading pre-trained model from {self.path}")
            model.load_state_dict(torch.load(self.path))
    
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
            loss = self._train_part(self.train_loader, self.model, optimizer, criterion, laman = self.laman)

            auc_test, acc_test, roc_test = self._test_part(self.test_loader, self.model, laman = self.laman)
            auc_val, acc_val, roc_val = self._test_part(self.val_loader, self.model, laman = self.laman)
                
            if (epoch)%5 == 0:
                auc_train, acc_train, roc_train = self._test_part(self.train_loader, self.model, laman = self.laman)
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Auc_train: {auc_train:.4f}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')
            else:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')

        time_end = time.time()
        print("--------------------------------")
        print()
        print(f"Time to train model for 1 epoch = {(time_end - time_start)/epochs} seconds")
        print()
        print()
        
        if self.save_model:
            path = f'/global/homes/d/dimathan/Laman-Graphs-and-Jets/{self.model_info["model_key"]}_p{self.input_dim}_{self.pair_input_dim}.pth'
            print(f"Saving model to {path}")
            torch.save(self.model.state_dict(), path) 
        
        return auc_test, roc_test

        
    #---------------------------------------------------------------
    def _train_part(self, train_loader, model, optimizer, criterion, laman = False):

        model.train() # Set model to training mode. This is necessary for dropout, batchnorm etc layers 
                                  # that behave differently in training mode vs eval mode (which is the default)
                                  # We need to include this since we have particlenet.eval() in the test_particlenet function
        
        loss_cum = 0              # Cumulative loss
        
        for index, data in enumerate(train_loader):
            inputs, labels = data[0], data[1]
            inputs = inputs.to(self.torch_device)
            labels = labels.to(self.torch_device) 
            if laman:
                graph = data[2].to(self.torch_device)
            else: 
                graph = None
                        
            # zero the parameter gradients
            optimizer.zero_grad()

            # create pt of each particle instead of (px, py, pz, E) for the input 
            pt = torch.sqrt(inputs[:, 0, :]**2 + inputs[:, 1, :]**2)
            pt = pt.unsqueeze(1).clamp(min=10**-8)

            # forward + backward + optimize
            outputs = model(x = pt if self.input_dim==1 else inputs, v = inputs, graph = graph)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_cum += loss.item()
            # Cache management
            torch.cuda.empty_cache()
        
        return loss_cum


    #---------------------------------------------------------------
    @torch.no_grad()
    def _test_part(self, test_loader, model, laman = False):
        model.eval()

        all_labels = []
        all_output_softmax = []

        for index, data in enumerate(test_loader):
            inputs, labels = data[0], data[1]
            inputs = inputs.to(self.torch_device)
            labels = labels.to(self.torch_device) 
            if laman:
                graph = data[2].to(self.torch_device)
            else: 
                graph = None


            # create pt of each particle instead of (px, py, pz, E) for the input 
            pt = torch.sqrt(inputs[:, 0, :]**2 + inputs[:, 1, :]**2)
            pt = pt.unsqueeze(1).clamp(min=10**-8)

            #if laman: # sort the particles by pt. Required for Laman Graphs which are constructed in the architecture.ParticleTransformer script
            #    pt, indices = torch.sort(pt, dim = 2, descending = True)
            #    # Gather inputs according to the sorted indices along the particles dimension
            #    inputs = torch.gather(inputs, dim=2, index=indices.expand_as(inputs))

            outputs = model(x = pt if self.input_dim==1 else inputs, v = inputs, graph = graph)

            output_softmax = torch.nn.functional.softmax(outputs, dim=1) # Keep on GPU
            all_output_softmax.append(output_softmax[:, 1].detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())


        # Calculate ROC, AUC outside the loop. Make lists of arrays into arrays
        all_labels = np.concatenate(all_labels)
        all_output_softmax = np.concatenate(all_output_softmax)
        
        accuracy = sklearn.metrics.accuracy_score(all_labels, all_output_softmax > 0.5)
        auc = sklearn.metrics.roc_auc_score(all_labels, all_output_softmax)
        roc = sklearn.metrics.roc_curve(all_labels, all_output_softmax)
        
        return (auc, accuracy, roc)