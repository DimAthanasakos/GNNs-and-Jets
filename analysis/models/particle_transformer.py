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
import sys
import glob

import matplotlib.pyplot as plt
import sklearn
import scipy

import torch
import networkx
import energyflow
from analysis.architectures import ParticleTransformer  
import random

from dataloader import read_file

def laman_graph(x): # For now this is not used 
    batch_size, _, seq_len = x.shape
    indices = np.zeros((batch_size, seq_len, seq_len))
    for i in range(seq_len-2):
        indices[:, i, i+1] = 1
        indices[:, i, i+2] = 1
    indices[seq_len-2, seq_len-1] = 1 
    
    return indices

def nearest_neighbors(x):
    x = torch.from_numpy(x) 

    batch_size, _, num_particles = x.size()
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)

    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    
    x = torch.cat((rapidity, phi), dim=1) # (batch_size, 2, num_points)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)                                    # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                               # (batch_size, num_points, num_points)

    # Make the upper right triangle of the distance matrix infinite so that we don't connect the i-th particle with the j-th particle if i > j 
    # This also avoids double-counting the same distance (ij and ji)
    pairwise_distance = torch.tril(pairwise_distance) - torch.triu(torch.ones_like(pairwise_distance)*float('inf'))  # -inf because topk indices return the biggest values -> we've made all distances negative 

    # Initialize a boolean mask with False (indicating no connection) for all pairs
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool)

    # The non-padded particles for each jet
    non_zero_particles = np.linalg.norm(x, axis=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)

    # Find the indices of the 2N-3 nearest neighbors for each jet and connect them: 
    for b in range(batch_size):
            if valid_n[b] <= 1:
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
            
            angles = 2*valid_n[b] - 3 # The maximum number of angles we can add until it becomes a fully connected graph
            
            bool_mask[b, row_indices[:angles], col_indices[:angles]] = True

    # Make the Laman Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    bool_mask = bool_mask.numpy() 

    return bool_mask 

def random_laman_graph(x): 
    batch_size, _, num_particles = x.shape
    # for each b in batch size, calculate the number of non-zero particles and permute them 
    non_zero_particles = np.linalg.norm(x, axis=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)
    idx = np.zeros((batch_size, num_particles, 2))
    # keep track of all perms so to remove the upper-triagonal part that stems from the first 3 particles forming a triangle
    perms = []
    for b in range(batch_size):
        if valid_n[b] <= 3:
            continue
        permutation = np.random.permutation(valid_n[b])
        perms.append(permutation)

        # connect the 3 first particles that are permuted
        idx[b, permutation[0] ] = [permutation[1], permutation[2]]
        idx[b, permutation[1] ] = [permutation[0], permutation[2]]
        idx[b, permutation[2] ] = [permutation[0], permutation[1]]
        # connect the rest of the particles in a Henneberg construction: Connect the i-th hardest particle with the 2 closest particles, i_1 and i_2, where i_1,2 < j
        for i in range(3, valid_n[b]):
            # for each particle i, add an edge to 2 particles j < i, at random 
            idx[b, permutation[i] ] = random.sample(list(permutation[:i]), 2)
        # fill the rest of the indices with valid_n[b] - 1, valid_n[b] - 2
        idx[b, valid_n[b]:] = [valid_n[b] - 1, valid_n[b] - 2]
    
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

    # ensure that the adjacency matrix is lower diagonal, useful for when we add angles later at random, to keep track of the connections we remove/already have
    #mask_upper = ~torch.triu(torch.ones(num_particles, num_particles, dtype=torch.bool), diagonal=0)
    #bool_mask = bool_mask & mask_upper.unsqueeze(0)

    # Remove some angles at random between the particles. Default value of angles = 0.
    #bool_mask = angles_laman(x, bool_mask, angles, pairwise_distance = pairwise_distance) 

    # Make the Laman Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    bool_mask = bool_mask.numpy() 

    return bool_mask 
        

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


def shannon_entropy(adjacency_matrices, x):
    batch_size, _, num_particles = x.shape
    non_zero_particles = np.linalg.norm(x, axis=1) != 0
    valid_n = non_zero_particles.sum(axis = 1) 

    valid_mask = np.arange(num_particles) < valid_n[:, None]
    
    # Use the mask to select elements from adjacency_matrices and sum to count neighbors
    n_neighbors = np.where(valid_mask[:, :, None], adjacency_matrices, 0).sum(axis=2)
    
    # Adjust n_neighbors based on valid_n, setting counts to 1 beyond valid particles
    # This step is no longer necessary as np.where and broadcasting handle the adjustment implicitly

    # Compute Shannon entropy
    # Avoid division by zero or log of zero by replacing non-valid n_neighbors with 1
    n_neighbors[n_neighbors == 0] = 1
    epsilon = 1e-8
    shannon_entropy_batch = np.log(n_neighbors).sum(axis=1) / ( valid_n + epsilon ) / np.log(valid_n - 1 + epsilon)
    shannon_entropy_batch = np.nan_to_num(shannon_entropy_batch)  # Handle divisions resulting in NaN
    shannon_entropy = np.mean(shannon_entropy_batch)
    print(f"Shannon Entropy = {shannon_entropy}")
    print()

    return shannon_entropy

def connected_components(adjacency_matrices, x):
    batch_size, _, num_particles = x.shape
    non_zero_particles = np.linalg.norm(x, axis=1) != 0
    valid_n = non_zero_particles.sum(axis = 1) 

    # use scipy.sparse.csgraph.connected_components to calculate the connected components for each graph in the batch
    connected_components = np.zeros((batch_size, num_particles))
    avg_n_components = 0
    for b in range(batch_size):
        adjacency_matrix = adjacency_matrices[b, :valid_n[b], :valid_n[b]]
        n_components, labels = scipy.sparse.csgraph.connected_components(adjacency_matrix, directed=False)
        avg_n_components += n_components
    # Average number of connected components
    avg_n_components = avg_n_components / batch_size
    
    print()
    print(f"Average number of connected components = {avg_n_components}")
    print()
    return 


# Create a knn Graph  
def knn(x, k, angles = 0, extra_info = False): 
    print()  
    print(f"Constructing a pure knn graph with k = {k}")
    print()
    x = torch.from_numpy(x) 

    batch_size, _, num_particles = x.size()

    non_zero_particles = np.linalg.norm(x, axis=1) != 0
    valid_n = non_zero_particles.sum(axis = 1)

    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)

    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = torch.atan2(py, px)
    
    x = torch.cat((rapidity, phi), dim=1) # (batch_size, 2, num_points)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)                                    # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                               # (batch_size, num_points, num_points)

    # Mask the diagonal 
    # Create a mask for the diagonal elements across all matrices in the batch
    eye_mask = torch.eye(num_particles, dtype=torch.bool).expand(batch_size, num_particles, num_particles)
    pairwise_distance[eye_mask] = -float('inf')

    # Mask all the padded particles, i.e. n >= valid_n[b]
    # Create an indices tensor
    indices = torch.arange(pairwise_distance.size(1), device=pairwise_distance.device).expand_as(pairwise_distance)

    valid_n_tensor = torch.tensor(valid_n, device=pairwise_distance.device).unsqueeze(1).unsqueeze(2)

    # Now you can use valid_n_tensor in your operation
    mask_row = indices >= valid_n_tensor
    mask_col = indices.transpose(-2, -1) >= valid_n_tensor

    # Apply the masks
    pairwise_distance[mask_row] = -float('inf')
    pairwise_distance[mask_col] = -float('inf')

    # Find the indices of the 2 nearest neighbors for each particle        
    idx = pairwise_distance.topk(k=k, dim=-1) # It returns two things: values, indices 
    idx = idx[1] # (batch_size, num_points, 2)
    
    # Initialize a boolean mask with False (indicating no connection) for all pairs
    bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool)

    # Efficiently populate the boolean mask based on laman_indices
    for i in range(k):  # Assuming each particle is connected to two others as per laman_indices
        # Extract the current set of indices indicating connections
        current_indices = idx[:, :, i]

        # Generate a batch and source particle indices to accompany current_indices for scatter_
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_particles)
        src_particle_indices = torch.arange(num_particles).expand(batch_size, -1)

        # Use scatter_ to update the bool_mask; setting the connection locations to True
        bool_mask[batch_indices, src_particle_indices, current_indices] = True

    # Make the Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    av = 0
    for b in range(batch_size):
        edges = torch.sum(bool_mask[b, :valid_n[b], :valid_n[b]], dim=(0,1)).item() / 2
        av += edges / (2*valid_n[b]-3) 
    av = av / batch_size
    print(f"Average number of edges/2n-3 = {av}")
    print()

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    bool_mask = bool_mask.numpy() 

    # Calculate the Shannon Entropy and the number of connected components
    if extra_info:
        connected_components(bool_mask, x)
        shannon_entropy(bool_mask, x)

    return bool_mask 

# Create a Laman Graph using a mod of the k nearest neighbors algorithm.
def laman_knn(x, angles = 0, extra_info = False):   
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

    # ensure that the adjacency matrix is lower diagonal, useful for when we add angles later at random, to keep track of the connections we remove/already have
    mask_upper = ~torch.triu(torch.ones(num_particles, num_particles, dtype=torch.bool), diagonal=0)
    bool_mask = bool_mask & mask_upper.unsqueeze(0)

    # Remove some angles at random between the particles. Default value of angles = 0.
    bool_mask = angles_laman(x, bool_mask, angles, pairwise_distance = pairwise_distance) 

    # Make the Laman Edges bidirectional 
    bool_mask = bool_mask | bool_mask.transpose(1, 2)

    # transform to numpy. That's because we later transform everything on the dataset to pytorch 
    # TODO: Change this code to work with numpy from the start
    bool_mask = bool_mask.numpy() 

    if extra_info:
        # Calculate the Shannon Entropy and the number of connected components
        connected_components(bool_mask, x)
        shannon_entropy(bool_mask, x)

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
        self.classification_task = model_info['classification_task']
        if self.classification_task not in ['ZvsQCD', 'qvsg']: 
            sys.exit('Invalid classification task. Choose between ZvsQCD and qvsg. For the potential extension to other tasks, please check dataloader.py and modify the code accordingly.')
        
        self.n_total = model_info['n_total']
        self.n_train = model_info['n_train']
        self.n_test = model_info['n_test']
        self.n_val = model_info['n_val'] 
        self.batch_size = self.model_info['model_settings']['batch_size']
        
        self.load_model = model_info['model_settings']['load_model'] # Load a pre-trained model or not
        self.save_model = model_info['model_settings']['save_model'] # Save the model or not
      
        self.input_dim = model_info['model_settings']['input_dim'] # 4 for (px, py, pz, E) as input for each particle, 1 for pt
        if self.input_dim not in [1, 4]:
            raise ValueError('Invalid input_dim at the config file for ParT. Must be 1 or 4') 
        
        self.pair_input_dim = model_info['model_settings']['pair_input_dim']  # how many interaction terms for pair of particles. 
                                                                              # If 3: use (dR, k_t = min(pt_1, pt_2)*dR, z = min(pt_1, pt_2)/(pt_1 + pt_2) ),
                                                                              # if 4: also use m^2 = (E1 + E2)^2 - (p1 + p2)^2    

        if model_info['model_key'].endswith('graph'): # Graph-Transfomer, e.g. Laman Graph or a KNN Graph
            self.graph_transformer = True
            self.graph_type = self.model_info['model_settings']['graph']
            self.add_angles = model_info['model_settings']['add_angles']
            if self.graph_type == 'knn_graph': self.k = model_info['model_settings']['k']
            else:  self.k = None
            if self.graph_type == 'laman_knn_graph': self.sorting_key = model_info['model_settings']['sorting_key']
            else:  self.sorting_key = 'pt'
                
        else:                                         # Vanilla Particle Transformer
            self.graph_transformer = False
            self.graph_type = None
            self.add_angles = 0


        #if self.graph_transformer and self.input_dim == 4: 
        #    raise ValueError('Invalid input_dim at the config file for ParT. Must be 1 for Laman Graphs')
            
        self.train_loader, self.val_loader, self.test_loader = self.init_data()

        self.model = self.init_model()

    #---------------------------------------------------------------
    def init_data(self):
        # Choose the dataset to load
        # The jetclass dataset origin: https://github.com/jet-universe/particle_transformer 
        # It has 10 different classes of jets, each class has 10M jets. 
        # Currently we are using the Z vs QCD dataset. For more details on the classes of jets look at the dataloader.py script and the 
        # github repository mentioned above.

        if self.classification_task == 'ZvsQCD':
            # Each file contains 100k jets for each class
            nz = nqcd = self.n_total // 2 

            directory_path = '/pscratch/sd/d/dimathan/JetClass_Dataset/'
            Z_jet_filepattern=  f"{directory_path}/ZToQQ*"
            QCD_jet_filepattern = f"{directory_path}/ZJetsToNuNu*"
            # read all files with those patterns in '/pscratch/sd/d/dimathan/JetClass_Dataset/'
            # Getting the list of files that match the patterns
            Z_jet_files = glob.glob(Z_jet_filepattern)
            QCD_jet_files = glob.glob(QCD_jet_filepattern)

            print()
            print(f"Found {len(Z_jet_files)} files matching ZToQQ pattern.")
            print(f"Found {len(QCD_jet_files)} files matching ZJetsToNuNu pattern.")
            print()

            x_particles_Z, x_jet_Z, y_Z = np.array([]), np.array([]), np.array([]) 
            for file in Z_jet_files:
                x_particles, x_jet, y = read_file(filepath = file, labels = ['label_Zqq', 'label_QCD'])
                x_particles_Z = np.concatenate((x_particles_Z, x_particles), axis = 0) if x_particles_Z.size else x_particles
                x_jet_Z = np.concatenate((x_jet_Z, x_jet), axis = 0) if x_jet_Z.size else x_jet
                y_Z = np.concatenate((y_Z, y), axis = 0) if y_Z.size else y
                if x_particles_Z.shape[0] >= nz: 
                    x_particles_Z = x_particles_Z[:nz]
                    x_jet_Z = x_jet_Z[:nz]
                    y_Z = y_Z[:nz]
                    break # Stop reading files if we have enough jets
            
            x_particles_qcd, x_jet_qcd, y_qcd = np.array([]), np.array([]), np.array([])
            for file in QCD_jet_files:
                x_particles, x_jet, y = read_file(filepath = file, labels = ['label_Zqq', 'label_QCD'])
                x_particles_qcd = np.concatenate((x_particles_qcd, x_particles), axis = 0) if x_particles_qcd.size else x_particles
                x_jet_qcd = np.concatenate((x_jet_qcd, x_jet), axis = 0) if x_jet_qcd.size else x_jet
                y_qcd = np.concatenate((y_qcd, y), axis = 0) if y_qcd.size else y
                if x_particles_qcd.shape[0] >= nqcd: 
                    x_particles_qcd = x_particles_qcd[:nqcd]
                    x_jet_qcd = x_jet_qcd[:nqcd]
                    y_qcd = y_qcd[:nqcd]
                    break

            # concatenate the two datasets 
            self.X_ParT = np.concatenate((x_particles_Z, x_particles_qcd), axis = 0)
            self.Y_ParT = np.concatenate((y_Z, y_qcd), axis = 0)

            # print how many jets we've loaded 
            print()
            print(f"Loaded {self.X_ParT.shape[0]} jets for the Z vs QCD classification task.")
            print()
            self.Y_ParT = self.Y_ParT[:, 0] # one-hot encoding, where 0: Background (QCD) and 1: Signal (Z) 
            # match the shape of the data to the shape of the energyflow data for consistency
            self.X_ParT = np.transpose(self.X_ParT, (0, 2, 1))

        elif self.classification_task == 'qvsg': 
            # Load the four-vectors directly from the quark vs gluon data set
            self.X_ParT, self.Y_ParT = energyflow.datasets.qg_jets.load(num_data=self.n_total, pad=True, 
                                                            generator='pythia',  # Herwig is also available
                                                            with_bc=False        # Turn on to enable heavy quarks
                                                        )                        # X_PFN.shape = (n_jets, n_particles per jet, n_variables)  

        # Preprocess by centering jets and normalizing pts
        for x_ParT in self.X_ParT:
            mask = x_ParT[:,0] > 0
            yphi_avg = np.average(x_ParT[mask,1:3], weights=x_ParT[mask,0], axis=0)
            x_ParT[mask,1:3] -= yphi_avg
            x_ParT[mask,0] /= x_ParT[:,0].sum()

        # Delete the last-column (pid or masses or E) of the particles
        self.X_ParT = self.X_ParT[:,:,:3]

        # TODO:
        # Change the architecture.ParticleTransformer script to accept (pt, eta, phi) as input features for the interaction terms instead of (px, py, pz, E) in order to save compute time
        # The input terms for each particle are left as given in the ParticleTransformer architecture.
            
        # Change the order of the features from (pt, eta, phi, pid) to (px, py, pz, E) to agree with the architecture.ParticleTransformer script
        self.X_ParT = energyflow.p4s_from_ptyphims(self.X_ParT)

        # (E, px, py, pz) -> (px, py, pz, E)
        self.X_ParT[:,:, [0, 1, 2, 3]] = self.X_ParT[:,:, [1, 2, 3, 0]] 
        
        # Transpose the data to match the ParticleNet/ParT architecture convention which is (batch_size, n_features, n_particles) 
        # instead of the current shape (batch_size, n_particles, n_features)
        self.X_ParT = np.transpose(self.X_ParT, (0, 2, 1))

        train_loader, val_loader, test_loader = self.load_data(self.X_ParT, self.Y_ParT, graph_transformer = self.graph_transformer, sorting_key = self.sorting_key)

        return train_loader, val_loader, test_loader
    

    def load_data(self, X, Y, graph_transformer = False, sorting_key = None):
        ''' 
        Split the data into training, validation and test sets depending on the specifics of the model.
        '''
        if graph_transformer:
            print()
            print(f"Sorting the particles based on the {sorting_key} key.")
            # Sort the particles based on the sorting key
            if sorting_key in ['angularity_increasing', 'angularity_decreasing']:
                px, py, pz, energy = X[:, 0:1, :], X[:, 1:2, :], X[:, 2:3, :], X[:, 3:4, :]
                
                rapidity = 0.5 * np.log(1 + (2 * pz) / np.clip(energy - pz, a_min=1e-10, a_max=None))
                phi = np.arctan2(py, px)
                dr = np.sqrt(rapidity**2 + phi**2)
                
                decreasing = True if sorting_key == 'angularity_decreasing' else False 
                sorting_condition =  (X[:, 0:1, :]**2 + X[:, 1:2, :]**2)*dr * (-1 if decreasing else +1)  # The zero-padded particles will have sorting_condition = 0
                
                if not decreasing: 
                    # we have to be careful with the zero-padded particles. We want them to be at the end of the sorted array 
                    condition = np.linalg.norm(X, axis=1) == 0
                    condition = condition[:, np.newaxis, :]
                    sorting_condition = np.where(condition, float("inf"), sorting_condition)
                    
                sorted_indices = np.argsort( sorting_condition , axis=-1)

            # Sort by 'pt' (also the default option)
            else:
                # we need to sort based on pt for the Laman Graphs 
                sorted_indices = np.argsort( -(X[:, 0, :]**2 + X[:, 1, :]**2), axis=-1)[:, np.newaxis, :]

            X = np.take_along_axis(X, sorted_indices, axis=-1)

            t_st = time.time()
            
            # We need to constuct the graph in chunks to avoid memory issues when n_total > 10^6
            chunk_size = 20*1024  # Adjust this based on your memory constraints and the size of self.X_ParT
            total_size = X.shape[0]  # Assuming the first dimension is the batch size
            chunks = (total_size - 1) // chunk_size + 1  # Calculate how many chunks are needed

            if self.graph_type == 'laman_random_graph': 
                graph = np.concatenate([random_laman_graph(X[i * chunk_size:(i + 1) * chunk_size]) for i in range(chunks)] )
            
            elif self.graph_type == 'laman_knn_graph': 
                graph = np.concatenate([laman_knn(X[i * chunk_size:(i + 1) * chunk_size], angles = self.add_angles, extra_info=True if i==0 else False) for i in range(chunks)])

            elif self.graph_type == '2n3_nearest_neighbors': 
                graph = np.concatenate([nearest_neighbors(X[i * chunk_size:(i + 1) * chunk_size]) for i in range(chunks)] ) 
                
            elif self.graph_type == 'knn_graph':
                k = self.k
                graph = np.concatenate([knn(X[i * chunk_size:(i + 1) * chunk_size], k = k, extra_info=True if i==0 else False) for i in range(chunks)] )
                
            else: 
                sys.exit("Invalid graph type for Laman Graphs. Choose between 'laman_random_graph', 'laman_knn_graph, '2n3_nearest_neighbors' and 'knn_graph'") 

            print(f"Time to create the graph = {time.time() - t_st} seconds")

            (features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test, 
            graph_train, graph_val, graph_test) = energyflow.utils.data_split(X, Y, graph,
                                                                              val=self.n_val, test=self.n_test, shuffle = True)

            # Data loader   
        
            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_train).float(), torch.from_numpy(Y_ParT_train).long(), torch.from_numpy(graph_train).bool() )
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

            val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_val).float(), torch.from_numpy(Y_ParT_val).long(), torch.from_numpy(graph_val).bool() )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size, shuffle=True)
            
            test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features_test).float(), torch.from_numpy(Y_ParT_test).long(), torch.from_numpy(graph_test).bool() ) 
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True)

        # For the case of Vanilla Particle Transformer
        else: 
            (features_train, features_val, features_test, Y_ParT_train, Y_ParT_val, Y_ParT_test) = energyflow.utils.data_split(X, Y,
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
            self.path = f'/global/homes/d/dimathan/Laman-Graphs-and-Jets/Saved_Model_weights/{self.model_info["model_key"]}_p{self.input_dim}_{self.pair_input_dim}.pth'
            print(f"Loading pre-trained model from {self.path}")
            model.load_state_dict(torch.load(self.path, map_location=self.torch_device))
    
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

        best_auc_test = 0
        best_auc_val, best_roc_val = None, None 

        for epoch in range(1, epochs+1):
            print("--------------------------------")
            loss = self._train_part(self.train_loader, self.model, optimizer, criterion, graph_transformer = self.graph_transformer)

            auc_test, acc_test, roc_test = self._test_part(self.test_loader, self.model, graph_transformer = self.graph_transformer)
            auc_val, acc_val, roc_val = self._test_part(self.val_loader, self.model, graph_transformer = self.graph_transformer)
            
            # Save the model with the best test AUC
            if auc_test > best_auc_test:
                best_auc_test = auc_test
                best_auc_val = auc_val
                best_roc_val = roc_val
                # store the model with the best test AUC
                if self.save_model:
                    best_model_params = self.model.state_dict()

            if (epoch)%5 == 0:
                auc_train, acc_train, roc_train = self._test_part(self.train_loader, self.model, graph_transformer = self.graph_transformer)
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Auc_train: {auc_train:.4f}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')
            else:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}, Test Acc: {acc_test:.4f}, Test AUC: {auc_test:.4f}')

        time_end = time.time()
        print("--------------------------------")
        print()
        print(f"Time to train model for 1 epoch = {(time_end - time_start)/epochs:.1f} seconds")
        print()
        print(f"Best AUC on the test set = {best_auc_test:.4f}")
        print(f"Corresponding AUC on the validation set = {best_auc_val:.4f}")
        print()
        if self.save_model:
            path = f'/global/homes/d/dimathan/Laman-Graphs-and-Jets/Saved_Model_weights/{self.model_info["model_key"]}_p{self.input_dim}_{self.pair_input_dim}.pth'
            print(f"Saving model to {path}")
            torch.save(best_model_params, path) 
        
        return best_auc_val, best_roc_val

        
    #---------------------------------------------------------------
    def _train_part(self, train_loader, model, optimizer, criterion, graph_transformer = False):

        model.train() # Set model to training mode. This is necessary for dropout, batchnorm etc layers 
                                  # that behave differently in training mode vs eval mode (which is the default)
                                  # We need to include this since we have particlenet.eval() in the test_particlenet function
        
        loss_cum = 0              # Cumulative loss
        
        for index, data in enumerate(train_loader):
            inputs, labels = data[0], data[1]
            inputs = inputs.to(self.torch_device)
            labels = labels.to(self.torch_device) 
            
            if graph_transformer:
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
    def _test_part(self, test_loader, model, graph_transformer = False):
        model.eval()

        all_labels = []
        all_output_softmax = []

        for index, data in enumerate(test_loader):
            inputs, labels = data[0], data[1]
            inputs = inputs.to(self.torch_device)
            labels = labels.to(self.torch_device) 
            
            if graph_transformer:
                graph = data[2].to(self.torch_device)
            else: 
                graph = None

            # create pt of each particle instead of (px, py, pz, E) for the input 
            pt = torch.sqrt(inputs[:, 0, :]**2 + inputs[:, 1, :]**2)
            pt = pt.unsqueeze(1).clamp(min=10**-8)

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