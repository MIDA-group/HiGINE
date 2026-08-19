#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shallow-learning baselines (SVC, k-NN, logistic regression) for survival
classification from patient-level aggregated cell features, for comparison
against HiGINE.
"""

import os
import argparse
import yaml
from types import SimpleNamespace
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
import matplotlib

font = {'size': 14}
matplotlib.rc('font', **font)


# load params
params_file = './settings/mif/shallowlearning_baseline.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--parameters-file', default=params_file, type=str, help='file where the parameters for the baselines are stored')
args = parser.parse_args()
params_file = args.parameters_file

with open(params_file, 'r') as params:
    p = SimpleNamespace(**yaml.safe_load(params))

FOLDS_PATH = p.FOLDS_PATH
FOLDS_FORMAT = p.FOLDS_FORMAT
FOLDS = p.FOLDS
path_feats = p.features_file
path_label = p.LABEL_DATA
FEATURE_LIST = p.FEATURE_LIST
use_stage_as_feature = p.use_stage_as_feature
stages0 = p.STAGE_CLASS_0

# how to show (or not show) plots
if p.SHOW_PLOTS:
    if p.BLOCK_ON_PLOTS:
        showplot=lambda:plt.show()
    else:
        showplot=lambda:plt.pause(0.001)
else:
    showplot=lambda:plt.close()

# whether cell-level features include a tissue-category column: if so, features
# are aggregated separately per tissue category (mean/std), otherwise they are
# summed across all cells in the sample
SPLIT_BY_TISSUE = 'Tissue Category' in FEATURE_LIST


# load labels

df_l = pd.read_csv(path_label)

y_all = pd.DataFrame(columns=['lbl', 'cls'])
for i in set(df_l.ID):
    y_all.loc[i, 'lbl'] = df_l.loc[df_l.ID == i, 'Follow-up (days)'].values
    y_all.loc[i, 'cls'] = df_l.loc[df_l.ID == i, 'label'].values
    y_all.loc[i, 'dead'] = df_l.loc[df_l.ID == i, 'Dead/Alive'].values == 'Dead'


# load features

df = pd.read_csv(path_feats)

SAMPLE_ID_COLUMN = p.SAMPLE_ID_COLUMN

first = True
for index, my_id in enumerate(set(df[SAMPLE_ID_COLUMN])):
    my_sample = df.loc[df[SAMPLE_ID_COLUMN] == my_id]

    # the ID column is a plain number in some datasets (e.g. the public BOMI
    # release) and a "... # N" string in others (e.g. IMC)
    raw_id = my_sample.ID.iloc[0]
    i = int(raw_id.split(' # ')[1]) if isinstance(raw_id, str) else int(raw_id)
    if i not in list(df_l.ID):
        print('skipped: ', i)
        continue

    sample_feats = my_sample[FEATURE_LIST]

    if SPLIT_BY_TISSUE:
        samplet = sample_feats.loc[sample_feats['Tissue Category'] == 'Tumor']
        samplet = samplet.drop(columns='Tissue Category')
        samplemt = samplet.mean()
        samplest = samplet.std()
        samples = sample_feats.loc[sample_feats['Tissue Category'] == 'Stroma']
        samples = samples.drop(columns='Tissue Category')
        samplems = samples.mean()
        sampless = samples.std()

        samplemt.index = ['Tumorm_' + i for i in samplemt.index]
        samplems.index = ['Stromam_' + i for i in samplems.index]
        samplest.index = ['Tumors_' + i for i in samplest.index]
        sampless.index = ['Stromas_' + i for i in sampless.index]
        sample = pd.concat([samplemt, samplems, samplest, sampless])
    else:
        samplesum = sample_feats.sum()
        samplesum.index = ['sum_' + i for i in samplesum.index]
        sample = samplesum

    if use_stage_as_feature:
        sample['Stage'] = 0 if df_l.loc[df_l.ID == i, p.STAGE_COLUMN].values[0] in stages0 else 1

    if first:
        x_all = pd.DataFrame(columns=sample.index)
        first = False

    x_all.loc[i] = sample
    print(index, my_id, end='\r')

x_all = x_all.loc[y_all.index] # remove wrong rows


# all
cindex = {'SVC': [], '5-NN': [], 'logreg': []}
auc = {'SVC': [], '5-NN': [], 'logreg': []}

for fold_num in range(FOLDS):
    id_test = pd.read_csv(os.path.join(FOLDS_PATH, FOLDS_FORMAT.format(fold_num, 'test') + '.csv'))
    id_val = pd.read_csv(os.path.join(FOLDS_PATH, FOLDS_FORMAT.format(fold_num, 'val') + '.csv'))
    id_train = pd.read_csv(os.path.join(FOLDS_PATH, FOLDS_FORMAT.format(fold_num, 'train') + '.csv'))

    X_train = pd.concat([x_all.loc[id_train.ID], x_all.loc[id_val.ID]])
    X_test = x_all.loc[id_test.ID]
    y_train = pd.concat([y_all.loc[id_train.ID, 'cls'], y_all.loc[id_val.ID, 'cls']]).to_list()
    y_test = y_all.loc[id_test.ID, 'cls'].to_list()
    y_event_time = y_all.loc[id_test.ID, 'lbl'].to_list()
    y_event = y_all.loc[id_test.ID, 'dead'].to_list()

    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', probability=True)).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    cindex['SVC'].append(concordance_index_censored(y_event, y_event_time, [i[0] for i in y_pred]))
    auc['SVC'].append(roc_auc_score(y_test, [i[1] for i in y_pred]))

    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(5, weights='distance')).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    cindex['5-NN'].append(concordance_index_censored(y_event, y_event_time, [i[0] for i in y_pred]))
    auc['5-NN'].append(roc_auc_score(y_test, [i[1] for i in y_pred]))

    clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    cindex['logreg'].append(concordance_index_censored(y_event, y_event_time, [i[0] for i in y_pred]))
    auc['logreg'].append(roc_auc_score(y_test, [i[1] for i in y_pred]))


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 5), sharey=True)

ax.plot([0.5, 3.5], [0.5, 0.5], 'grey', linestyle='--')
x = [[j[0] for j in cindex[k]] for k in cindex]
lbls = list(cindex)
ax.boxplot(x, tick_labels=lbls, notch=False)
ax.set_ylabel('c-index')
ax.grid()
ax.set_ylim((0.0, 1.0))

showplot()


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 5), sharey=True)

ax.plot([0.5, 3.5], [0.5, 0.5], 'grey', linestyle='--')
x = [auc[k] for k in auc]
lbls = list(auc)
ax.boxplot(x, tick_labels=lbls, notch=False)
ax.set_ylabel('AUC')
ax.grid()
ax.set_ylim((0.0, 1.0))

showplot()


print('c-index')
for k in cindex:
    print(k)
    x = [j[0] for j in cindex[k]]
    print(np.mean(x))
    print(np.std(x))

print()
print('auc')
for k in auc:
    print(k)
    x = auc[k]
    print(np.mean(x))
    print(np.std(x))
