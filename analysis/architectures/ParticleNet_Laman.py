''' 
This file contains the ParticleNet model. This is taken (and modified) from the implementation in https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleNet.py
'''


''' ParticleNet

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570

Adapted from the DGCNN implementation in https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.
'''
import numpy as np
import torch
import torch.nn as nn
import time 
import sys

# Base class
sys.path.append('.')

# Find the k nearest neighbors for each point. For the first layer we use the (η,φ) coordinates of the particles as input. For the next layers we use the output of the previous layer. 
def knn(x, k, Laman = False):                                                          # x: (batch_size, num_dims, num_points)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)                                    # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                               # (batch_size, num_points, num_points)
    
    #print(f"knn")
    #print(f"Laman = {Laman}")
    #print(f"k={k}") 
    if not Laman: # Usual knn algorithm 
        # Find the indices of the k nearest neighbors
        idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]                     # (batch_size, num_points, k)

    elif Laman: # Laman Construction. For now we only support Henneberg construction where the i-th particle in a jet is connected with the 2 closest particles, that are harder than it, so with l, k s.t. l, k < i.
      
        k = 2 # Laman graph has k = 2
        
        # Connect the 3 hardest particles in the jet in a triangle 
        
        idx_3 = pairwise_distance[:, :3, :3].topk(k=3, dim=-1) # (batch_size, 3, 2)
        idx_3 = [idx_3[0][:,:,1:], idx_3[1][:,:,1:]] # (batch_size, 3, 1)

        # Connect the rest of the particles in a Henneberg construction: Connect the i-th hardest particle with the 2 closest particles, i_1 and i_2, where i_1,2 < j  
        pairwise_distance = pairwise_distance[:, 3:, :] # Remove the pairwise distances of 3 hardest particles from the distance matrix 
    
        # Make the upper right triangle of the distance matrix infinite so that we don't connect the i-th particle with the j-th particle if i > j 
        pairwise_distance = torch.tril(pairwise_distance, diagonal=2) - torch.triu(torch.ones_like(pairwise_distance)*float('inf'), diagonal=3)  # -inf because topk indices return the biggest values -> we've made all distances negative 

        # Find the indices of the 2 nearest neighbors for each particle
        
        idx = pairwise_distance.topk(k=2, dim=-1) # (batch_size, num_points, 2)
        
        idx = idx[1] # (batch_size, num_points, 2)
        
        # Concatenate idx and idx_3 to get the indices of the 3 hardest particles and the 2 nearest neighbors for the rest of the particles
        idx = torch.cat((idx_3[1], idx), dim=1) # (batch_size, num_points, 3)

    return idx



# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx, Laman = False):

    # We want to get the graph features before we pass them through the EdgeConvBlock.
    # For each particle in the graph, we use its features and the features of its k nearest neighbors as input to the EdgeConvBlock.
    # We could use (?) a GraphConvolutionalNetwork with directed edges where the edges are the k nearest neighbors. 
    # For some reason, ParticleNet uses a Conv2D layer instead. So we need to reshape the feature space to be able to use it.
    # We concatenate the features of each particle with the difference between its features and that of its k nearest neighbors. 
    # So now each particles' feature space "knows" about its k nearest neighbors.
    # For a nn.Conv2d layer, the input shape is (batch_size, num_channels, height, width). 
    # In our case, the input will be: (batch_size, 2*num_dims, num_points, k) where num_dims = num_features and num_points = num_particles in a jet.
    # So the height of the input is num_points and the width is k. 

    # We transform x ->  (batch_size, num_dims, num_points, k) 
    # So x[0, :, 0, :] is the first particle in the first jet. The 2nd dim (dim=1) corresponds to its features, so (pt, eta, phi, ...). 
    # The last dim (dim=3) is the same features but repeated k times.
    # As such: x[0, :, 0, n] = x[0, :, 0, m] for all n, m. 

    # fts: (batch_size, num_dims, num_points, k) 
    # fts[0, :, 0, :] corresponds the first particle in the first jet. The 2nd dim (dim=1) corresponds to features, so (pt, eta, phi, ...). 
    # The last dim (dim=3) corresponds to it's i-th nearest neighbor features 
    # so lets say that the 0th hadron has the 1st hadron as its nearest neighbor. Then fts[0, :, 0, 0] = x[0, :, 1, 0].
    
    # In the end, we do the following operation: fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k) 
    # So now fts[0, :num_dims, :, :] is the same as x[0, :, :, :] and 
    # fts[0, num_dims:, 0, :] is the difference of the features of the hardest particle and its k nearest neighbors. 


    # For example, for fts = [0, :, 0, 0] . This corresponds to the "top-left" pixel in the input image for the first jet. 
    # The features of this pixel is: 2*num_dims dimensional. The top half of the vector are the input features of the first/hardest particle in the jet (so pt, η, φ, ...)
    # The bottom half of the vector are the differences of the features of the first particle (so pt, η, φ, ...) and its nearest neighbor (singular). 
    # The next pixel is fts = [0, :, 0, 1].  The top half are the same as before.
    # The bottom half of the vector are the differences of the features of the first particle and its 2nd nearest neighbor, and so on.
    # The pixel at fts = [0, :, n , m] corresponds the (n+1)-th hardest particle in the jet and its (m+1)-th nearest neighbor.
    
    #print(f"get_graph_feature_v1: Laman = {Laman}")
    if not Laman:
        batch_size, num_dims, num_points = x.size()                         # num_dims = num_features, num_points = num_particles in a jet 
        
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        # Initially we have x: (batch_size, num_dims, num_points) and idx: (batch_size, num_points, k) 
        
        fts = x.transpose(2, 1).reshape(-1, num_dims)                       # (batch_size, num_dims, num_points) -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
       
        fts = fts[idx, :].view(batch_size, num_points, k, num_dims)         # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
        
        fts = fts.permute(0, 3, 1, 2).contiguous()                          # (batch_size, num_dims, num_points, k)
    
        x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)  # (batch_size, num_dims, num_points) -> (batch_size, num_dims, num_points, 1) -> (batch_size, num_dims, num_points, k)
                                                                            # So we repeat the features of each particle k times. This covers a row of size k in the input image.
                                                                            # Every element in that row is the same.
                                                                            
        fts = torch.cat((x, fts - x), dim=1)                                # ->(batch_size, 2*num_dims, num_points, k)
                                                                            # Think of this as an image where the first half of the channels are the features of the particles, 
                                                                            # The rows correspond to the different particles. All the k columns in a row are the same. 
                                                                            # The second half of the channels are the differences of the features of the particles and their neighbors. 
                                                                            # These columns are NOT repeated k times.
        
    
    elif Laman:
        batch_size, num_dims, num_points = x.size()                         # num_dims = num_features, num_points = num_particles in a jet 
        
        
        if num_dims == 7:
            idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            #print(f"idx_base.shape = {idx_base.shape}")
            #print(f"idx.shape = {idx.shape}")
            #print(f"idx[0, :5, :] = {idx[0, :5, :]}")
            idx = idx.view(-1)

            # Initially we have x: (batch_size, num_dims, num_points) and idx: (batch_size, num_points, k) 
            
            fts = x.transpose(2, 1).reshape(-1, num_dims)                       # (batch_size, num_dims, num_points) -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
        
            fts = fts[idx, :].view(batch_size, num_points, k, num_dims)         # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
            
            fts = fts.permute(0, 3, 1, 2).contiguous()                          # (batch_size, num_dims, num_points, k)
        
            x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)  # (batch_size, num_dims, num_points) -> (batch_size, num_dims, num_points, 1) -> (batch_size, num_dims, num_points, k)
                                                                                # So we repeat the features of each particle k times. This covers a row of size k in the input image.
                                                                                # Every element in that row is the same.
            fts_temp = fts.clone()                                                       
            fts = torch.cat((x, fts - x), dim=1)                                # ->(batch_size, 2*num_dims, num_points, k)
                                                                                # Think of this as an image where the first half of the channels are the features of the particles, 
                                                                                # The rows correspond to the different particles. All the k columns in a row are the same. 
                                                                                # The second half of the channels are the differences of the features of the particles and their neighbors. 
                                                                                # These columns are NOT repeated k times.
        
            # print the fts and x tensors of the first jet in the batch
            #print(f"idx.shape = {idx.shape}")
            #print(f"idx = {idx}")   
            #print(f"x.shape = {x.shape}")
            #print(f"fts_temp.shape = {fts_temp.shape}")
            #print(f"fts.shape = {fts.shape}")
            #print(f"x[0, :, :5, :] = {x[0, :, :5, :]}")
            #print(f"fts_temp[0, :, :5, :] = {fts_temp[0, :, :5, :]}")
            #print(f"fts[0, 7:, :5, :] = {fts[0, 7:, :5, :]}")      
            #print()

        elif num_dims == 3:

            idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
            
            idx = idx + idx_base

            #print(f"idx_base.shape = {idx_base.shape}")
            #print(f"idx.shape = {idx.shape}")
            #print(f"idx[0, :5, :] = {idx[0, :5, :]}")
            
            idx = idx.view(-1)
            
            # For the 3-dim repres. of the Laman Graph:
            # We want the features to be shaped: (batch_size, num_features = 3, num_points, k=2)
            # k = 2 because this is the Laman graph
            # num_features = 3 because the features will be: (pt, δpt, δR) 
            # So the pt features will look like : [[p_T^0, p_T^0], [p_T^1, p_T^1],  [p_T^2, p_T^2]...]
            # δpt features will look like : [[p_T^0 - p_T^1, p_T^0-p_T^2], [p_T^1-p_T^0, p_T^1-p_T^2],  [p_T^2-p_T^0, p_T^2-p_T^1]...]
            # δR features will look like : [[δR^01, δR^02], [δR^10, δR^12],  [δR^20, δR^21]...] 
            
            pt_features = x[:,2:,:]
            
            pt_features = pt_features.view(batch_size, 1, num_points, 1).repeat(1, 1, 1, k)
            
            fts = x.transpose(2, 1).reshape(-1, num_dims)                       # (batch_size, num_dims, num_points) -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
        
            fts = fts[idx, :].view(batch_size, num_points, k, num_dims)         # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
            
            fts = fts.permute(0, 3, 1, 2).contiguous()                          # (batch_size, num_dims, num_points, k)
        
            pt_features_neighbors = fts[:,2:,:,:]
            
            x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)  # (batch_size, num_dims, num_points) -> (batch_size, num_dims, num_points, 1) -> (batch_size, num_dims, num_points, k)

            fts_temp = fts.clone()
            d_features = fts - x 

            dr_features = torch.sum(torch.pow(d_features[:,0:1,:,:],2) + torch.pow(d_features[:,1:2,:,:],2), dim = 1, keepdim=True)

            #print(f"d_features.shape = {d_features.shape}")
            #print(f"d_features[0, :, :4, :] = {d_features[0, :, :4, :]}")
            #print()
            #print(f"dr_features.shape = {dr_features.shape}")
            #print(f"dr_features[0, :, :4, :] = {dr_features[0, :, :4, :]}")
            #print()
            dr_features_sqrt = torch.sqrt(dr_features + 1e-9) # Add a small number to avoid division by zero 


            #dr_features = torch.sqrt(torch.sum(d_features[:,0:2,:,:]**2, dim=1, keepdim=True),  requires_grad=True)
            
            #print(f"d_features.shape = {d_features.shape}")
            #print(f"dr_features_sqrt.shape = {dr_features_sqrt.shape}")
            #print(f"dr_features_sqrt[0, :, :4, :] = {dr_features_sqrt[0, :, :4, :]}")
            #print()
            
            #fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
            
            #fts = torch.sqrt(fts[:, :3, :, :])
            #fts = torch.cat((pt_features, d_features[:, 2:3, :, :], dr_features), dim=1)  # ->(batch_size, 3, num_points, k)
            #fts[:, 2:3, :, :] = torch.sqrt(fts[:, 2:3, :, :])
            
            fts = torch.cat((pt_features, d_features[:, 2:3, :, :], dr_features_sqrt), dim=1)
            
            
            
            # print the fts and x tensors of the first jet in the batch            
            #print(f"x.shape = {x.shape}")
            #print(f"d_features.shape = {d_features.shape}")
            #print(f"dr_features.shape = {dr_features.shape}")
            #print(f"fts.shape = {fts.shape}")
            #print(f"fts_temp.shape = {fts_temp.shape}")
            #print(f"x[0, :, :, :] = {x[0, :, :4, :]}")
            #print(f"fts_temp[0, :, :4, :] = {fts_temp[0, :, :4, :]}")
            #print(f"d_features[0, :, :, :] = {d_features[0, :, :4, :]}")
            #print(f"dr_features[0, :, :, :] = {dr_features[0, :, :4, :]}")
            #print(f"fts_temp[0, :, :5, :] = {fts_temp[0, :, :5, :]}")
            #print(f"fts[0, :, :4, :] = {fts[0, :, :4, :]}")      
            #print()

            #time.sleep(10)

        
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False, Laman=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        # For the first layer of the edgeconv block, we need to concatenate the original features with the difference between itself's and the neighbors' features
        # So the input dimension of the first layer is 2 * in_feat. For the rest of the layers, the input dimension is the same as the output dimension of the previous layer
        # Because we do NOT use k-nn again. 
        # For the input of the FIRST EDGE CONV LAYER, we pass the laman graph so the input dimension is 3 (pt, δpt, δR)

        self.convs = nn.ModuleList() # nn.ModuleList allows us to use the modules in a Python list 
        for i in range(self.num_layers):
            if Laman and i == 0 and in_feat == 3:
                input_dims = 3
            elif i == 0:
                input_dims = 2 * in_feat
            else:
                input_dims = out_feats[i - 1]
            self.convs.append(nn.Conv2d(input_dims, out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm: # perform batch normalization which normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation: 
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        # We add a shortcut (or skip) connection to the output of the last convolutional layer. Check if the input and output dimensions are the same. If not, perform a convolution to match the dimensions.
        if in_feat == out_feats[-1]: # to check if the input and output dimensions are the same. 
            self.sc = None # if they are the same, we don't need to perform a convolution in order to add a shortcut connection
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features, Laman = False):

        topk_indices = knn(points, self.k, Laman = Laman)
        x = self.get_graph_feature(features, self.k, topk_indices, Laman = Laman)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        # After the three conv2d layers: x.shape = (batch_size, C_out, num_particles, k) -> (batch_size, C_out, num_particles). The aggregation is done by taking the mean of the k neighbors for each feature of each particle.
        fts = x.mean(dim=-1)  # (N, C, P)

        # We add a shortcut (or skip) connection to the output of the last convolutional layer
        if self.sc: # Runs if the input and output dimensions are not the same -> perform a convolution to match the dimensions
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:       # Input and output dimensions are the same -> no need to perform a convolution 
            sc = features

        #print(f"The output of EdgeConvBlock: {self.sc_act(sc + fts).shape}")        
        #print()
        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):
    # the default parameters are the same as the original ParticleNet-Lite 
    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(2, (32, 32, 32)), (7, (64, 64, 64))], # Two EdgeConv layers. The first layer has k=2 because it corresponds to the Laman graph. The second layer has k=7 because it corresponds to the k-nn graph.
                 fc_params=[(128, 0.1)],                             # One fully connected layer with 128 output channels and a dropout rate of 0.1
                 Laman = False,
                 use_fusion=True,                                    
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,                                
                 for_segmentation=False,                            
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)
        # Print the conv and fc parameters 
        print()
        print(f"conv_params = {conv_params}")
        print(f"fc_params = {fc_params}")
        print()
        print(f"Laman = {Laman}")
        self.Laman = Laman
        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param 
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1] # the input dim of the edge conv block is the output dim of the previous edge conv block expect 
                                                                              # for the first edge conv block where the input dim is the input dim of the ParticleNet

            # Note: For the first layer, if we use the Laman graph, the neighbors (k) has to be k=2. 
            # Check knn() and get_graph_feature_v1() for more details.                                                             
            self.edge_convs.append(EdgeConvBlock(k = 2 if (idx == 0 and self.Laman) else k, in_feat = in_feat, out_feats = channels, cpu_mode = for_inference, Laman = True if (idx == 0 and self.Laman) else False))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #print(f"features.shape: {features.shape}")
        #features = features[:, :5, :]
        #points = points[:, :5, :]
        
        # The ParticleNet code wants the inputs to have the shape: (njets, nfeatures, nparticles) But our convention so far has been: (njets, nparticles, nfeatures)
        points = points.transpose(1, 2).to(device) 
        features = features.transpose(1, 2).to(device) 
        
        # If we have not defined a particular mask, the default one is: Mask all points that have a zero feature vector (in case of padding the input) 
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (njets, 1, hadrons) 
            
        points *= mask
        features *= mask

        # Since we'll use the points to calculate the distance between the points, 
        # we need to shift the coordinates of the particles that are the result of zero-padding \
        # to infinity so that they don't affect the distance calculation. 
        coord_shift = (mask == 0) * 1e9 # shape (njets, 1, hadrons)
        
        if self.use_counts:
            counts = mask.float().sum(dim=-1) # mask contains boolean True = 1 and False = 0. Summing over the last 
                                              # dimension gives the number of True values for each jet, 
                                              # i.e. the number of hadrons
            counts = torch.max(counts, torch.ones_like(counts))  # >=1
            # the shape is (njets, 1, 1)
        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features

        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            if idx == 0:
                # We pass fts after we do a BatchNorm. So the δR in fts and in pts are NOT the same. Is this a problem ???   
                pts = points + coord_shift                          # for the first layer use points=(η,φ) for the kNN to construct the Laman Graphs. 
                fts = conv(pts, fts, Laman = self.Laman) * mask
            else:
                pts = fts + coord_shift                             # For the following layers use the output of the previous layer
                                                                    # So the full features at the latent space are used for the kNN at all layers except the first one
                fts = conv(pts, fts) * mask

            if self.use_fusion:
                outputs.append(fts)
            #print(f"idx = {idx}, fts.shape = {fts.shape}")
            #print(f"outputs = {outputs}")
            #print()

        # Global Average Pooling
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts: # We "average" over all non-zero particles in the jet. This is a permutation invariant operation on a equivariant quantity (fts) -> x is permutation invariant
                                # (batch_size, channels, nparticles) -> (batch_size, channels) 
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

                
        # Final Fully connected layers. Remember that x is permutation invariant -> output is permutation invariant
        output = self.fc(x)

        if self.for_inference:
            output = torch.softmax(output, dim=1) # shape (njets, num_classes) 
            
        return output

