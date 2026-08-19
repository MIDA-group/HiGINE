#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partition point clouds into subgraph clusters for each core.
"""

import os
import logging
import json
import yaml
import argparse
from types import SimpleNamespace
from shutil import copyfile
import numpy as np
import pandas as pd
from multiprocessing import Pool


logging.basicConfig(level=logging.INFO)

# load params
params_file = './settings/mif/subsample.yaml'

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
    logging.info(f'New path for results was created: {p.results_path}')
else:
    logging.error(f'Results path exists already: {p.results_path}')
    raise FileExistsError(f'Folder: {p.results_path} exists already')

# copy parameters into folder
copyfile(params_file, os.path.join(p.results_path, 'subsample_PARAMS.yaml'))


# read sample info
logging.info(f'reading sample info from: {p.features_file}')
df = pd.read_csv(p.features_file)
sample_names = list(dict.fromkeys(df[p.SAMPLE_ID_COLUMN])) #dict.fromkeys() instead of set() to preserve order

x_name_df = 'Cell X Position'
y_name_df = 'Cell Y Position'


def task(sample_index, sample_name):
    """Partition one sample's point cloud into spatially defined subgraph clusters.

    Args:
        sample_index: Index of the sample in the global sample list, used to assign
            globally unique cluster IDs (offset = sample_index * CLUSTERS_PER_SAMPLE).
        sample_name: Sample name string used to filter rows from the global dataframe.

    Returns:
        Dict mapping cluster ID strings to lists of cell row indices.
    """
    # Calculate starting cluster number for this sample
    cluster_num_start = sample_index * p.CLUSTERS_PER_SAMPLE

    sample = df.loc[df[p.SAMPLE_ID_COLUMN] == sample_name]
    raw_id = sample.ID.iloc[0]
    if isinstance(raw_id, float) and np.isnan(raw_id):
        logging.info(f'Sample {sample_name} has no ID, skipping')
        return None
    # the ID column is a plain number in some datasets (e.g. the public BOMI
    # release) and a "... # N" string in others (e.g. IMC)
    ID = int(raw_id.split('# ')[1]) if isinstance(raw_id, str) else int(raw_id)

    # local cluster index (offset by global start)
    cluster_num = cluster_num_start
    clusters = {}

    def split(sample):
        """Partition the sample into overlapping rectangular clusters using a sliding window.

        Scans left to right in x and top to bottom in y. A cluster is recorded whenever
        at least `min_num_nodes` cells fall within the current rectangle. Results are
        written into the outer `clusters` dict.
        """
        nonlocal cluster_num

        # define borders
        sample_xmin = sample[x_name_df].min()
        sample_xmax = sample[x_name_df].max()
        sample_ymin = sample[y_name_df].min()
        sample_ymax = sample[y_name_df].max()

        # splitting $totalWidth$ into $x$ number of columns, with a optimal column width of $width$ and a overlap of $b$: $totalWidth = x*(width-b) + b$ with $totalWidth = xmax - xmin$
        x_steps = max(1, int(round((sample_xmax - sample_xmin - p.subsample_x_overlap) / (p.average_x_stepsize - p.subsample_x_overlap), 0))) # number of step in x direction, at least one
        # taking the rounded value to get the exact $width$
        x_intervals = (sample_xmax - sample_xmin + (x_steps-1) * p.subsample_x_overlap) / x_steps
        logging.debug(f'actual width: {x_intervals}')

        # init xmin
        xmin = sample_xmin

        # for all x_columns
        for x_sub_num in range(x_steps):
            # init coordinates
            xmax = xmin + x_intervals
            yold = sample_ymin
            ymax = sample_ymin

            # while y within the y_rows
            y_sub_num = 0
            while sample_ymin <= ymax <= sample_ymax:
                # init y coordinates
                ymin = yold
                ymax = yold

                # for improved performance a subsample is used
                subsample = sample.loc[(sample[x_name_df] > xmin) & (sample[x_name_df] < xmax) & (sample[y_name_df] > ymin)]
                # current number of nodes in the sample
                n = 0
                # move ymax until desired number of nodes is within rectangle
                while n <= p.nodes_per_subsample:
                    if n >= p.nodes_per_subsample - p.subsample_y_overlap and ymin == yold:
                        yold = ymax
                    ymax += p.y_step_size
                    n = len(subsample.loc[sample[y_name_df] < ymax])
                    if ymax > sample_ymax:
                        break

                if n > p.min_num_nodes:
                    # hierarchical graph id
                    hid = y_sub_num % 2 * 2 + x_sub_num % 2

                    clusters['ID__{}__samplename__{}__clusternum__{}__hID__{}'.format(ID, sample_name, cluster_num, hid)] = \
                        subsample.loc[(xmin < subsample[x_name_df]) & (subsample[x_name_df] < xmax) &\
                                    (ymin < subsample[y_name_df]) & (subsample[y_name_df] < ymax)].index.to_list()
                    cluster_num += 1
                y_sub_num += 1
            xmin = xmax - p.subsample_x_overlap


    split(sample)

    num_clusters_in_sample = cluster_num - cluster_num_start
    logging.info(f'Sample: {sample_name} ({sample_index:3d}/{len(sample_names)}), ID: {ID} -> {num_clusters_in_sample:4d} clusters (global IDs: {cluster_num_start}-{cluster_num-1})')

    if num_clusters_in_sample + 100 > p.CLUSTERS_PER_SAMPLE:
        if num_clusters_in_sample >= p.CLUSTERS_PER_SAMPLE:
            logging.warning('WARNING! Surpassing CLUSTERS_PER_SAMPLE number of clusters!')
        else:
            logging.warning('WARNING! Approaching CLUSTERS_PER_SAMPLE number of clusters!')

    return clusters


if __name__ == '__main__':
    clusters = {}
    with Pool() as pool:
        logging.info(f'Pool size: {pool._processes}')
        items = [(i, sn) for i, sn in enumerate(sample_names)]
        for result in pool.starmap(task, items):
            if result is not None:
                clusters |= result

    print(f'Created {len(clusters)} clusters in total')

    # sort to make output reproducible
    with open(os.path.join(p.results_path, "index_output.json"), "w") as outfile:
        json.dump(dict(sorted(clusters.items())), outfile)
