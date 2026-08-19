# -*- coding: utf-8 -*-
"""
HiGINE graph neural network model definition.
"""

import torch
from torch import nn
from torch_geometric.nn import GINConv, GINEConv, TopKPooling, \
    global_mean_pool, global_max_pool, SAGPooling, GATConv
from torch.nn import ModuleList
from torch_geometric.utils import dropout_node, dropout_edge, mask_select, to_undirected


class GNN(torch.nn.Module):
    """Graph neural network used in HiGINE.

    Stacks multiple conv+pool layers with optional skip connections,
    then passes the aggregated graph embedding through fully connected layers.

    Args:
        num_features: Number of input node features.
        num_class: Number of output classes.
        conv_nhid: Hidden channel sizes for each conv layer.
        gnn_pool: Pooling type after each conv layer ('topk' or 'sag').
        gnn_pool_ratio: Pooling ratio for each layer.
        lin_nhid: Hidden sizes for the fully connected layers.
        lin_dropout: Dropout probabilities for each FC layer.
        skip_connection: If True, concatenate readout from each conv layer.
        bn_momentum: Momentum for BatchNorm layers.
        drop_node: Node dropout probability per conv layer.
        drop_edge: Edge dropout probability per conv layer.
        gnn_name: Conv layer type per layer ('gine', 'gin', or 'gat').
        readout_name: Graph readout ('gmp', 'gmaxp', or 'max+mean').
        device: Torch device string.
    """

    def __init__(self, num_features, num_class, conv_nhid=[20, 40, 80], gnn_pool='topk',
                 gnn_pool_ratio=[0.5, 0.5, 0.5], lin_nhid=[40, 20], lin_dropout=[0.0],
                 skip_connection=True, bn_momentum=0.01, drop_node=[0.0, 0.0, 0.0], drop_edge=[0.0, 0.0, 0.0],
                 gnn_name=['GINE', 'GINE', 'GINE'], readout_name='gmp', device='cpu'):
        super(GNN, self).__init__()
        self.device = device
        self.skip_connection = skip_connection
        self.drop_node = drop_node
        self.drop_edge = drop_edge
        self.gnn_name = gnn_name

        if readout_name.lower() == 'gmp':
            self.readout = global_mean_pool
            lin_factor = 1
        elif readout_name.lower() == 'gmaxp':
            self.readout = global_max_pool
            lin_factor = 1
        elif readout_name.lower() == 'max+mean':
            self.readout = [global_mean_pool, global_max_pool]
            lin_factor = 2
        else:
            raise ValueError(f"Unknown readout_name: '{readout_name}'. Choose from 'gmp', 'gmaxp', 'max+mean'.")

        self.num_conv_layers = len(conv_nhid)

        # feature extraction GNN
        conv_nhid_tmp = [num_features] + conv_nhid
        self.conv = []
        for i, name in zip(range(len(conv_nhid)), gnn_name):
            if name.lower() == 'gine':
                self.conv.append(GINEConv(nn.Sequential(nn.Linear(conv_nhid_tmp[i], conv_nhid_tmp[i+1]),
                                               nn.BatchNorm1d(conv_nhid_tmp[i+1],momentum=bn_momentum), nn.ReLU(),
                                               nn.Linear(conv_nhid_tmp[i+1], conv_nhid_tmp[i+1]), nn.ReLU()),
                                 eps=0.0, train_eps=False, edge_dim=1))
            elif name.lower() == 'gin':
                self.conv.append(GINConv(nn.Sequential(nn.Linear(conv_nhid_tmp[i], conv_nhid_tmp[i+1]),
                                               nn.BatchNorm1d(conv_nhid_tmp[i+1],momentum=bn_momentum), nn.ReLU(),
                                               nn.Linear(conv_nhid_tmp[i+1], conv_nhid_tmp[i+1]), nn.ReLU()),
                                 eps=0.0, train_eps=False))
            elif name.lower() == 'gat':
                self.conv.append(GATConv(conv_nhid_tmp[i], conv_nhid_tmp[i+1]))
            else:
                raise ValueError(f"Unknown gnn_name: '{name}'. Choose from 'gine', 'gin', 'gat'.")
        self.conv = ModuleList(self.conv)

        # pooling layers
        if gnn_pool.lower() == 'topk':
            pool_func = TopKPooling
        elif gnn_pool.lower() == 'sag':
            pool_func = SAGPooling
        else:
            raise ValueError(f"Unknown gnn_pool: '{gnn_pool}'. Choose from 'topk', 'sag'.")

        self.pool = ModuleList([pool_func(conv_nhid[i], ratio=gnn_pool_ratio[i]) for i in range(len(conv_nhid))])

        # concat
        if self.skip_connection:
            self.lin_in = nn.Linear(sum(conv_nhid)*lin_factor, lin_nhid[0])
        else:
            self.lin_in = nn.Linear(conv_nhid[-1]*lin_factor, lin_nhid[0])

        # fully connected
        self.lin_dropout = ModuleList([nn.Dropout(p=p) for p in lin_dropout])
        self.hlin_list = ModuleList([nn.Linear(lin_nhid[i], lin_nhid[i+1]) for i in range(len(lin_nhid)-1)])
        self.lin_out = nn.Linear(lin_nhid[-1], num_class)

    def forward(self, data):
        x, edge_index, edge_attr, batch, lbl = data.x, data.edge_index, data.edge_attr, data.batch, data.y

        if len(edge_attr.shape) <= 1:  # GINE needs the additional dimension
            edge_attr = edge_attr[:, None]

        h_out = torch.empty((len(lbl), 0)).to(self.device)
        for ilayer, (dn, de, conv_layer, pool_layer) in enumerate(zip(self.drop_node, self.drop_edge, self.conv, self.pool)):
            # node/edge dropout
            edge_index, edge_mask, node_mask = dropout_node(edge_index, p=dn, num_nodes=len(x), relabel_nodes=True, training=self.training)
            x = mask_select(x, dim=0, mask=node_mask)
            edge_attr = mask_select(edge_attr, dim=0, mask=edge_mask)
            batch = mask_select(batch, dim=0, mask=node_mask)
            edge_index, edge_mask = dropout_edge(edge_index, p=de, training=self.training)
            edge_attr = mask_select(edge_attr, dim=0, mask=edge_mask)
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce='mean')

            # conv + pool
            if self.gnn_name[ilayer].lower() == 'gin':
                x = conv_layer(x, edge_index)
            else:
                x = conv_layer(x, edge_index, edge_attr)
            x, edge_index, edge_attr, batch, _, _ = pool_layer(x, edge_index, edge_attr, batch)

            if self.skip_connection:
                if not type(self.readout) == list:
                    h_out = torch.cat((h_out, self.readout(x, batch)), 1)
                else:
                    h_out = torch.cat((h_out, self.readout[0](x, batch), self.readout[1](x, batch)), 1)

        if not self.skip_connection:
            if not type(self.readout) == list:
                h_out = self.readout(x, batch)
            else:
                h_out = torch.cat((self.readout[0](x, batch), self.readout[1](x, batch)), 1)

        # fully connected layers
        hlin = self.lin_in(h_out)
        hlin = self.lin_dropout[0](hlin)
        for i_layer, layer in enumerate(self.hlin_list):
            hlin = layer(hlin)
            hlin = self.lin_dropout[i_layer+1](hlin)
        y = self.lin_out(hlin)
        return y
