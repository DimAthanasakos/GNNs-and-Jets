''' Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py

In this version I have deleted the two extra classes: ParticleTransformerTagger, ParticleTransformerTaggerWithExtraPairFeatures
since we only care about the kinematic features of the particles.

'''
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import time

import warnings

def laman_graph(x): 
    batch_size, _, seq_len = x.size() 
    indices = np.zeros((batch_size, seq_len, seq_len))
    for i in range(seq_len-2):
        indices[:, i, i+1] = 1
        indices[:, i, i+2] = 1
    indices[seq_len-2, seq_len-1] = 1 
    
    return indices


# Create a Laman Graph using a mod of the k nearest neighbors algorithm.
def knn(x):                                                                            # x: (batch_size, (η, φ), num_points)

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
        
    idx = pairwise_distance.topk(k=2, dim=-1) # (batch_size, num_points, 2)
    
    idx = idx[1] # (batch_size, num_points, 2)
        
    # Concatenate idx and idx_3 to get the indices of the 3 hardest particles and the 2 nearest neighbors for the rest of the particles
    idx = torch.cat((idx_3[1], idx), dim=1) # (batch_size, num_points, 3)

    # Make this into a boolean tensor to use for masking later: 
    # Initialize a boolean mask with False, assuming no connections
    idx_mask = torch.zeros(batch_size, num_particles, num_particles,  dtype=torch.bool, device = idx.device)

    # Assuming laman_indices contains valid indices within [0, num_particles)
    # Update bool_mask to True for connections indicated in laman_indices

    if False:
        time_st = time.time()
        for b in range(batch_size):
            for p in range(num_particles):
                # Get indices for connections of particle p in batch b
                connections = idx[b, p]
                # Ensure connections are within bounds
                connections = connections[connections < num_particles]
                # Mark these connections as True
                idx_mask[b, p, connections] = True
        print(f"Time to create the mask = {time.time() - time_st}")

        return idx_mask
    else:
        return idx



@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part

# Transform the 4-momentum vector to the (pt, rapidity, phi, mass) representation
def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False): 
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)

# TODO:
# The current input to this functions requires x to be [px, py, pz, E] and then transformed to [pt, eta, phi]
# Our datasets by default produce [pt, eta, phi] so currently we do the redundant transformation: [pt, eta, phi] -> [px, py, pz, E] in models.ParticleTransformer 
# and then -> [pt, eta, phi] here: 
def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs). uu holds the values of the pairs whose indices are in idx. 
    # return: (N, C, seq_len, seq_len)
    # N = batch_size, C = num_fts 
     
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len) # ensures that all indices in idx refer to particles with index < seq_len  
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0), # tensor with size [1, batch_size * num_fts * num_pairs]
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0), # tensor with size [1, batch_size * num_fts * num_pairs]
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0), # row indices of the pairs in the sparse tensor rep 
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0), # col indices of the pairs in the sparse tensor rep
    ), dim=0) # tensor with size [4, batch_size * num_fts * num_pairs]
    
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len] # tensor with size [batch_size, num_fts, seq_len, seq_len]. It is a dense rep of the sparse tensor.


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for index, dim in enumerate(dims):
            module_list.extend([
                #nn.LayerNorm(input_dim) if input_dim > 1  else nn.Identity(), # LayerNorm averages across the feature space for the same particle. For 1d input space this leads to a random classifier.
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):

        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)

        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim # the number of features for the pairwise interaction terms. The default is 4 for [lnΔ, lnk_t, lnz, lnm^2]
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx) # partial: freezes the arguments of the function pairwise_lv_fts 
                                                                                                                 # to num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx
        self.out_dim = dims[-1] 

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')


    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len) with v_dim = num_features for the input to the interaction terms. The default is 4 for [px, py, pz, E]
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else: # at least one of x, uu is not None
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                # create the pairs of indices for the graph
                # For Transformer, we have a fully connected graph -> we need to create all pairs of indices
                # Careful to not double-count. 
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len) # (batch, v_dim, seq_len, seq_len) 
                    # print some values for x to see what it looks like
                    xi = x[:, :, i, j]  # (batch, v_dim, seq_len*(seq_len+-1)/2) +: if we include self-pairs, -: otherwise
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:  
                    # (batch, v_dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x, weights = self.attn(x, x, x, key_padding_mask=padding_mask,
                          attn_mask=attn_mask)  # (seq_len, batch, embed_dim). By default it returnes the attention weights as well 
            
            #print(f"residual.shape = {residual.shape}")
            #print(f"x.shape = {x.shape}")
            #print(f"x[0, :6, :6] = {x[0, :6, :6]}")
            #print(f"weights.shape = {weights.shape}")
            #print(f"weights[0, :6, :6] = {weights[0, :6, :6]}")
            # Pytorch throws a warning here which I suspect is relevant to: https://github.com/pytorch/pytorch/issues/95702 
            # It looks like a bug of pytorch 2.x 
            # TODO: Address this or hope that it's resolved in the next version of pytorch
            warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask is deprecated.*")

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        #print(f"x.shape = {x.shape}")

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        #print(f"x.shape = {x.shape}")
        #print()
        #time.sleep(4)

        return x


class ParticleTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4, # the default is [lnΔ, lnk_t, lnz, lnm^2] for each pair of particles 
                 pair_extra_dim=0, # ?
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],   # the MLP for transforming the particle features input 
                 pair_embed_dims=[64, 64, 64], # the MPL for transforming the pairwise features input, i.e. interactions. Note that later we add
                                               # one more layers to this to match the number of heads in the attention layer.
                 num_heads=8,  # how many attention heads in each particle attention block
                 num_layers=8, # how many particle attention blocks
                 num_cls_layers=2, # how many layers of attention for the "class attention block"
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],   # check this in relation to the cls token 
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        self.num_heads = num_heads

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
       # _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
       # _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim

        # Embed the particle features before passing them to the attention layers
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        
        # self.pair_embed is only used if we want "interaction terms" between pairs of particles. These act as bias in the attention layer.
        # It is the MLP that transforms the interactions before passing it to the attention layers 
        # The final embedding dim for the pair_embed is the same as the number of heads in the attention layer. 
        # Each head has only one bias feature for all pairs of particles.
        self.pair_embed = PairEmbed( 
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None, laman = False):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy] from which we construct the interaction terms between pairs of particles
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs): Sparce format for indexing the pairs  
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference: # if training 
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1)) # returns: (N, C', P, P)
            x, v, mask, uu = self.trimmer(x, v, mask, uu)            # the default is trim = False so this does nothing  
            padding_mask = ~mask.squeeze(1)                          # (N, P) and padded = 1, real particle = 0 now due to ~ (bitwise not)

        with torch.cuda.amp.autocast(enabled=self.use_amp): # if true it lowers the precision of some computations to half precision for faster computation
                                                            # The default for self.use_amp = False

            # input embedding 
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)  # masked_fill: fill the elements of x with 0 where mask is False
                                                                      # mask.permute(2, 0, 1) -> (P, N, 1)
          
            # pair embedding to get the interaction terms between pairs of particles -> Acts as the bias in the attention layer of the particle attn block.

            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            if laman: # filter the attn_mask with the laman_indices
                # Masking operation based on the Laman Graph.
                print('laman')
                laman_indices = knn(v) # pass (px, py, pz, E) to knn so that it can then construct the Laman Graph based on the mod. of the knn algo. 

                batch_size, num_particles, _ = laman_indices.shape
                # Initialize a boolean mask with False (indicating no connection) for all pairs
                bool_mask = torch.zeros((batch_size, num_particles, num_particles), dtype=torch.bool)

                # Efficiently populate the boolean mask based on laman_indices
                for i in range(2):  # Assuming each particle is connected to two others as per laman_indices
                    # Extract the current set of indices indicating connections
                    current_indices = laman_indices[:, :, i]

                    # Generate a batch and source particle indices to accompany current_indices for scatter_
                    batch_indices = torch.arange(batch_size, device=attn_mask.device).view(-1, 1).expand(-1, num_particles)
                    src_particle_indices = torch.arange(num_particles, device=attn_mask.device).expand(batch_size, -1)

                    # Use scatter_ to update the bool_mask; setting the connection locations to True
                    bool_mask[batch_indices, src_particle_indices, current_indices] = True

                #print()
                #print(f'laman_indices.shape: {laman_indices.shape}')
                #print(f"laman_indices[0, :7, :]: {laman_indices[0, :7, :]}")

                #print(f'bool_mask.shape: {bool_mask.shape}')
                #print(f"bool_mask[0, :7, :7]: {bool_mask[0, :7, :7]}")
                # Make the Laman Edges bidirectional 
                #bool_mask = bool_mask | bool_mask.transpose(1, 2)

                bool_mask =  bool_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).reshape(batch_size*self.num_heads, num_particles, num_particles).to(attn_mask.device)

                attn_mask_laman = torch.where(bool_mask, attn_mask, torch.tensor(0).to(attn_mask.dtype).to(attn_mask.device))

#                print(f'attn_mask.shape: {attn_mask.shape}')

 #               print(f'bool_mask.shape: {bool_mask.shape}')

  #              print(f"bool_mask[0, :7, :7]: {bool_mask[0, :7, :7]}")
     #           print()
   #             print(f"attn_mask[0, :7, :7]: {attn_mask[0, :7, :7]}")
    #            print()
      #          print(f"attn_mask_laman[0, :7, :7]: {attn_mask_laman[0, :7, :7]}")
       #         time.sleep(10)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask = attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output