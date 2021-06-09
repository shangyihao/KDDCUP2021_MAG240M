import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math
import numpy as np


class HgtConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, n_heads, num_relations, dropout=0.2, use_norm=True, use_RTE=False):
        super(HgtConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.use_RTE = use_RTE
        self.use_norm = use_norm
        self.n_heads = n_heads
        self.d_k = out_dim//n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(num_relations, n_heads))
        self.relation_att_weight = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))

        self.att_j = nn.Parameter(torch.Tensor(1, out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, out_dim))
        self.att_rel = nn.Parameter(torch.Tensor(1, num_types+num_relations+num_types))
        self.drop = nn.Dropout(dropout)

        if self.use_RTE:
            self.emb = RelTemporalEncoding(in_dim)

        self.reset_parameters()

    def reset_parameters(self):

        for t in range(self.num_types):
            glorot(self.k_linears[t].weight)
            glorot(self.q_linears[t].weight)
            glorot(self.v_linears[t].weight)
            glorot(self.a_linears[t].weight)
        glorot(self.relation_att_weight)
        glorot(self.relation_msg)
        glorot(self.att_j)
        glorot(self.att_r)
        glorot(self.att_rel)

    def forward(self, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time, edge_index):
        '''
                j: source, i: target; <j, i>
        '''
        h, c = self.n_heads, self.out_dim
        '''
        create relations one-hot
        '''
        x_j = self.k_linears[node_type_j](node_inp_j).view(-1, h, c)

        if self.use_RTE:
            x_i = self.k_linears[node_type_i](node_inp_i).view(-1, h, c)
            x_j = self.emb(x_j, edge_time)

        x_onehot = np.zeros(self.num_types+self.num_relations+self.num_types, dtype=int)
        x_onehot[node_type_j] = 1
        x_onehot[edge_type + self.num_types] = 1
        x_onehot[node_type_i + self.num_types + self.num_relations] = 1
        x_onehot = torch.from_numpy(x_onehot).to(node_inp_i.device)

        alpha_j = (x_j * self.att_j).sum(dim=-1)
        alpha_r = (x_i * self.att_r).sum(dim=-1)
        alpha_rel = (x_onehot * self.att_rel).sum(dim=-1)

        out = self.propagate(edge_index, x=(x_j, x_onehot, x_i),
                             alpha=(alpha_j, alpha_rel, alpha_r))


    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.emb(t))
