#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create subgraphs and core graphs from sampled point clouds.
"""

import os
import logging
import json
import yaml
import pickle
import argparse
from types import SimpleNamespace
from shutil import copyfile
import pandas as pd
import numpy as np
from multiprocessing import Pool
from utils.graph_utils import get_graph_data, get_hierarchical_graph


logging.basicConfig(level=logging.INFO)

# load params
params_file = './settings/mif/make_graphs.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--parameters-file', default=params_file, type=str, help='file where the parameters for the training are stored')
args = parser.parse_args()
params_file = args.parameters_file

logging.info(f'params_file: {params_file}')
with open(params_file, 'r') as params:
    p = SimpleNamespace(**yaml.safe_load(params))

# make output path
if not os.path.exists(p.results_path):
    os.makedirs(p.results_path)
    os.makedirs(os.path.join(p.results_path, 'h_files'))
    os.makedirs(os.path.join(p.results_path, 'files'))
    logging.info(f'New path for results was created: {p.results_path}')
else:
    logging.error(f'Results path exists already: {p.results_path}')
    raise FileExistsError(f'Folder: {p.results_path} exists already')

# copy parameters into folder
copyfile(params_file, os.path.join(p.results_path, 'graph_PARAMS.yaml'))
copyfile(os.path.join(p.subsamples_path, 'subsample_PARAMS.yaml'), os.path.join(p.results_path, 'subsample_PARAMS.yaml'))


# read sample info
df = pd.read_csv(p.features_file)

# read label info
df_l = pd.read_csv(p.labels_file)
id2label = dict(zip(df_l['ID'], df_l['label']))
logging.info('sample labels read successfully')

# normalize features
if p.NORMALIZE_FEATURES:
    logging.info('normalizing features')
    txt = []
    for i in p.FEATURE_LIST:
        if i == 'Tissue Category':
            df[i] = df[i].apply(lambda x: 1.0*(x=='Tumor'))
            txt.append(f'{i}_categorized = bool( {i} == \'Tumor\')\n')
        else:
            mn, mx = df[i].agg(['min', 'max'])  # faster than separate min(), max()
            if mx == mn:
                # zero-variance column (e.g. a cell-type category that never
                # occurs in the data): normalizing is undefined, leave as-is
                logging.warning(f'{i} has no variance (constant {mn}), leaving unnormalized')
                txt.append(f'{i}_normalized = {i}  # constant column, left unnormalized\n')
            else:
                txt.append(f'{i}_normalized = ( {i} - {mn} ) / ( {mx} - {mn} )\n')
                df[i] = df[i].apply(lambda x: (x-mn)/(mx-mn))

    with open(os.path.join(p.results_path, 'normalize.txt'), 'w') as f:
        f.writelines(txt)

    logging.info('features in feature_list have been normalized (between 0 and 1)')


# import clusters (created from 'subsample.py')
logging.info('reading clusters')
with open(os.path.join(p.subsamples_path, 'index_output.json'), "r") as outfile:
    clusters = json.load(outfile)

clusters_df = pd.DataFrame.from_dict(clusters, orient='index')
clusters_df = clusters_df.assign(ID = lambda x: [int(i.split('__')[1]) for i in x.index.to_list()])
uIDs = sorted(set(clusters_df.ID))  # unique IDs, in order

logging.info(f'{len(clusters_df.ID)} clusters with {len(uIDs)} unique IDs imported successfully')


def task(i, sn):
    """Build and save subgraphs and core graphs for one patient.

    Args:
        i: Index of the patient in the global ID list (used for progress logging).
        sn: Sample name / patient ID.
    """
    try:
        label = id2label[sn]
    except KeyError:
        logging.info(f'Lacking label for ID {sn}, skipping')
        return

    logging.info(f'Creating graphs for: {sn} ({i}/{len(uIDs)}), label: {label}')

    # fetch clusters of current sample
    split_clusters_df = clusters_df.loc[clusters_df.ID == sn]

    if split_clusters_df.empty:
        logging.warning('No clusters!')
        return
    else:
        logging.info(f'fetched {len(split_clusters_df)} clusters for {sn}')

    # put clusters in dict
    clusters_raw = {c: list(split_clusters_df.loc[c].values) for c in split_clusters_df.index}

    # build subgraph adjacency matrices and node features
    data, data2, _ = get_graph_data(df, clusters_raw, critical=p.CRITICAL, features_list=p.FEATURE_LIST)

    data['Id'] = sn
    data['Labels'] = label
    data['FullID'] = np.squeeze(split_clusters_df.index.values)

    outfile_sub = os.path.join(p.results_path, os.path.join('files', f'{sn}_Adj_Features'))
    with open(outfile_sub + '.dat', 'wb') as f:
        pickle.dump(data, f)

    # build core-level graph adjacency matrices
    pid_list = [i.split('__')[1][-4:] + i.split('__')[3] + '__' + i.split('__')[7] for i in data['FullID']]
    data_h, _ = get_hierarchical_graph(np.array(data2['CenterX']),
                                       np.array(data2['CenterY']),
                                       np.array(data2['NumNodes']),
                                       np.array(pid_list),
                                       np.squeeze(label),
                                       np.squeeze(split_clusters_df.index.values),
                                       critical=p.H_CRITICAL)

    outfile_h = os.path.join(p.results_path, os.path.join('h_files', f'{sn}_Adj_Features_hg'))
    with open(outfile_h + '.dat', 'wb') as f:
        pickle.dump(data_h, f)


if __name__ == '__main__':
    with Pool() as pool:
        items = [(i, sn) for i, sn in enumerate(uIDs)]
        pool.starmap(task, items)
