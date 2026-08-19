# -*- coding: utf-8 -*-
"""
Graph construction utilities for subgraphs and hierarchical core graphs.
"""

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List
from scipy.sparse import csr_matrix


@njit()
def _create_one_adj(x:list, y:list, critical:float):
    """Creates a weighted adjacency matrix from x, y coordinates."""
    num_node = len(x)
    A = np.zeros((num_node, num_node))
    for k in range(num_node):
        for j in range(k+1,num_node):
            dist = np.sqrt((x[k]-x[j])**2+(y[k]-y[j])**2)
            if 0 < dist < critical:
                A[k,j] = critical/dist
                A[j,k] = critical/dist
    return A


def get_hierarchical_graph(center_x, center_y, num_nodes, pid_list, lbl, fullid, critical=300):
    """Build hierarchical (core-level) graphs from subgraph centre coordinates.

    Args:
        center_x: Array of subgraph centre x-coordinates.
        center_y: Array of subgraph centre y-coordinates.
        num_nodes: Array of node counts per subgraph.
        pid_list: Array of position IDs grouping subgraphs into cores.
        lbl: Survival label shared by all subgraphs in this sample.
        fullid: Array of full subgraph ID strings.
        critical: Distance threshold for connecting two subgraph nodes.

    Returns:
        Tuple of (graph data dict, (coordsX, coordsY)).
    """
    pids = list(dict.fromkeys(pid_list))
    adj = [[] for _ in range(len(pids))]
    feature = [[] for _ in range(len(pids))]
    coordsX = [[] for _ in range(len(pids))]
    coordsY = [[] for _ in range(len(pids))]
    nodes = [[] for _ in range(len(pids))]
    lbls = [0 for _ in range(len(pids))]
    for num, pid in enumerate(pids):
        mask = np.where(pid_list==pid, True, False)
        x = center_x[mask]
        y = center_y[mask]
        x_typed, y_typed = List(), List()
        for j in x:
            x_typed.append(j)
        for j in y:
            y_typed.append(j)
        A = _create_one_adj(x_typed, y_typed, critical)
        adj[num] = csr_matrix(A)
        feature[num] = num_nodes[mask]
        coordsX[num] = x
        coordsY[num] = y
        lbls[num] = lbl
        nodes[num] = fullid[mask]

    return {'Adj':adj, 'Features':feature, 'IDs':pids, 'Labels':lbls, 'Nodes':nodes}, \
        (coordsX, coordsY)


def get_graph_data(df, cdf, critical, features_list=None):
    """Build subgraph adjacency matrices and node features for all clusters.

    Args:
        df: DataFrame of cell-level data with position and feature columns.
        cdf: Dict mapping cluster IDs to lists of cell indices.
        critical: Distance threshold for connecting two cells.
        features_list: List of feature column names to use as node features.
            Must be provided; raises ValueError if None.

    Returns:
        Tuple of (subgraph data dict, centre data dict, (coordsX, coordsY)).
    """
    if features_list is None:
        raise ValueError("features_list must be provided.")
    adj = [[] for _ in range(len(cdf))]
    feature = [[] for _ in range(len(cdf))]
    center_x = [0 for _ in range(len(cdf))]
    center_y = [0 for _ in range(len(cdf))]
    num_points = [0 for _ in range(len(cdf))]
    coordsX = [[] for _ in range(len(cdf))]
    coordsY = [[] for _ in range(len(cdf))]
    for num, i in enumerate(list(cdf.keys())):
        tmp = np.array(cdf[i])
        tmp = tmp[~np.isnan(tmp)]
        tmp_df = pd.concat([df.loc[int(j)] for j in tmp], axis=1).T

        x = tmp_df['Cell X Position'].values.astype(float).tolist()
        x_typed = List()
        for j in x:
            x_typed.append(j)
        y = tmp_df['Cell Y Position'].values.astype(float).tolist()
        y_typed = List()
        for j in y:
            y_typed.append(j)
        A = _create_one_adj(x_typed, y_typed, critical)
        A = csr_matrix(A)
        adj[num] = A
        feature[num] = tmp_df[features_list].values.astype(float).tolist()
        center_x[num] = np.mean(x)
        center_y[num] = np.mean(y)
        num_points[num] = len(x)
        coordsX[num] = x
        coordsY[num] = y

    return {'Adj':adj, 'Features':feature}, {'CenterX':center_x, 'CenterY':center_y, 'NumNodes':num_points}, (coordsX, coordsY)

