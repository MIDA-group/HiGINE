#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train model to predict survival based on coregraphs, with features from the subgraph model.
"""


import os
import sys
import gc
import logging
import yaml
from types import SimpleNamespace
from shutil import copyfile
import pandas as pd
import numpy as np
from collections import defaultdict
import csv
import argparse
import random #Only to set seed


logging.basicConfig(level=logging.INFO)


# set parameters
params_file = './settings/mif/train_coregraphs.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--parameters-file', default=params_file, type=str, help='file where the parameters for the training are stored')
parser.add_argument('--stdin-overrides', action='store_true', default=False, help='override parameter settings from stdin')
parser.add_argument('--results-path', default=None, type=str, help='override parameter file and stdin setting')
args = parser.parse_args()
params_file = args.parameters_file

logging.info(f'params_file: {params_file}')
with open(params_file, 'r') as params:
    dct = yaml.safe_load(params)
    p = SimpleNamespace(**dct)

# override
if args.stdin_overrides:
    dct = yaml.safe_load(sys.stdin.read())
    #It turns out, that trying to load invalid YAML data returns a str
    try:
        q = SimpleNamespace(**dct)
    except TypeError:
        logging.error('Incorrect YAML from stdin (did you forget space after colon?)')
        logging.info(dct)
        raise
    p.__dict__.update(q.__dict__)


# prepend base_path
p.results_path = os.path.join(p.base_path, p.results_path)

if args.results_path:
    p.results_path = args.results_path

p.MODEL_FILE = os.path.join(p.base_path, p.MODEL_FILE)

# if we already have subgraph predictions
if hasattr(p, 'PRED_PATH') and p.PRED_PATH:
    p.PRED_PATH = os.path.join(p.base_path, p.PRED_PATH)
else:
    p.PRED_PATH = None


# Must be set before any import torch
if hasattr(p, 'CUDA_DEVICE'):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p.CUDA_DEVICE)
# Determinisitic CUDA (https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torchmetrics
from torch_geometric.loader import DataLoader
from utils.training_utils import train_one_epoch, epoch_validation, evaluate, \
    extract_features, save_preds, load_data_subgraphs, my_mean, FeatureNoise, \
    load_data_h_graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))


# preprocessing parameters used
graphs_input_folder = p.graphs_path
with open(os.path.join(graphs_input_folder, p.GRAPH_PARAMS), 'r') as f:
    pre_p = SimpleNamespace(**yaml.safe_load(f))

# TODO: set the num_features in some other way
num_o_features = len(pre_p.FEATURE_LIST)
num_classes = 2
del pre_p # not used elsewhere


# Feature count from subgraphs
with open(os.path.join(p.base_path, p.SUBTRAIN_PARAMS), 'r') as f:
    tmp_yaml = yaml.safe_load(f)
    load_kwargs = tmp_yaml['KWARGS']
    if p.LAYER_OF_SUB == 0:
        extra_features = 2
    else:
        extra_features = load_kwargs['lin_nhid'][p.LAYER_OF_SUB]


# Optional stage info as input to core graphs
if p.ADD_STAGE:
    if p.LABEL_DATA:
        df_l = pd.read_csv(p.LABEL_DATA)
        df_l = df_l.set_index('ID')
    else:
        raise Exception('No file found to load cancer stage.')
else:
    df_l = None


# Create output path
if not os.path.exists(p.results_path):
    os.makedirs(os.path.join(p.results_path, 'best'))
    logging.info(f'new path for results was created: {p.results_path}')
else:
    logging.warning(f'existing path is used to write results: {p.results_path}')
    raise Exception(f'existing path is used to write results: {p.results_path}')

# copy parameters into output folder
copyfile(os.path.join(graphs_input_folder, p.SUBSAMPLING_PARAMS), os.path.join(p.results_path, 'subsample_PARAMS.yaml'))
copyfile(os.path.join(graphs_input_folder, p.GRAPH_PARAMS), os.path.join(p.results_path, 'graph_PARAMS.yaml'))
copyfile(os.path.join(p.base_path, p.SUBTRAIN_PARAMS), os.path.join(p.results_path, 'subtrain_PARAMS.yaml'))
copyfile(params_file, os.path.join(p.results_path, 'coretrain_PARAMS.yaml'))


# set random seed (is this enough? see train_subgraphs)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED_i = 314
set_seed(SEED_i)


# metrics
def one_hot_acc(pred, lbl):
    pred_ = torch.argmax(pred, axis=1)
    lbl_ = torch.argmax(lbl, axis=1)
    return torchmetrics.functional.accuracy(pred_, lbl_, task="multiclass", num_classes=num_classes)

def one_hot_bacc(pred, lbl):
    pred_ = torch.argmax(pred, axis=1)
    lbl_ = torch.argmax(lbl, axis=1)
    return torchmetrics.functional.accuracy(pred_, lbl_, task="multiclass", num_classes=num_classes, average="macro")

average_precision_score = torchmetrics.classification.MulticlassAveragePrecision(num_classes=num_classes, average="weighted")
def average_precision(pred, lbl):
    lbl_ = torch.argmax(lbl, axis=1)
    return average_precision_score(pred, lbl_)

auroc = torchmetrics.classification.MulticlassAUROC(num_classes=num_classes)
def one_hot_auroc(pred, lbl):
    lbl_ = torch.argmax(lbl, axis=1)
    return auroc(pred, lbl_)

metrics = {'Acc':one_hot_acc,
            'Bacc':one_hot_bacc,
            'AUROC':one_hot_auroc,
            'AP':average_precision}

# create augmentation
match p.AUGMENTATION.lower():
    case 'feature_noise':
        augmentation = FeatureNoise(p.SIGMA)
    case _:
        augmentation = None
        logging.info('No augmentation used')

logging.info(f'{augmentation} as augmentation used')



for fold_num in range(p.FOLDS):
    print(f'\nFOLD {fold_num} with SEED {SEED_i}',flush=True)
    # load sub sample files and h files

    h_extra_features = extra_features + 1 if p.ADD_STAGE else extra_features

    # test dataloader
    ds_name = p.folds_format.format(fold_num,'test')
    id_test = pd.read_csv(os.path.join(p.folds_path, f'{ds_name}.csv'))

    ds_test = []
    for id_ in [i[0] for i in id_test.values]:
        path = os.path.join(graphs_input_folder,  os.path.join('files', f'{id_}_Adj_Features.dat'))
        ds_test.extend(load_data_subgraphs(path, num_classes))
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False)

    dsh_test = []
    for id_ in [i[0] for i in id_test.values]:
        path = os.path.join(graphs_input_folder,  os.path.join('h_files', f'{id_}_Adj_Features_hg.dat'))
        dsh_test.extend(load_data_h_graphs(path, h_extra_features, num_classes, id_split=p.ID_SPLIT))

    # val dataloader
    ds_name = p.folds_format.format(fold_num,'val')
    id_val = pd.read_csv(os.path.join(p.folds_path, f'{ds_name}.csv'))

    ds_val = []
    for id_ in [i[0] for i in id_val.values]:
        path = os.path.join(graphs_input_folder,  os.path.join('files', f'{id_}_Adj_Features.dat'))
        ds_val.extend(load_data_subgraphs(path, num_classes))
    val_loader = DataLoader(ds_val, batch_size=p.BS, shuffle=False)

    dsh_val = []
    for id_ in [i[0] for i in id_val.values]:
        path = os.path.join(graphs_input_folder,  os.path.join('h_files', f'{id_}_Adj_Features_hg.dat'))
        dsh_val.extend(load_data_h_graphs(path, h_extra_features, num_classes, id_split=p.ID_SPLIT))

    # train dataloader
    ds_name = p.folds_format.format(fold_num,'train')
    id_train = pd.read_csv(os.path.join(p.folds_path, f'{ds_name}.csv'))

    ds_train = []
    for id_ in [i[0] for i in id_train.values]:
        path = os.path.join(graphs_input_folder,  os.path.join('files', f'{id_}_Adj_Features.dat'))
        ds_train.extend(load_data_subgraphs(path, num_classes))
    train_loader = DataLoader(ds_train, batch_size=p.BS, shuffle=False)

    dsh_train = []
    for id_ in [i[0] for i in id_train.values]:
        path = os.path.join(graphs_input_folder,  os.path.join('h_files', f'{id_}_Adj_Features_hg.dat'))
        dsh_train.extend(load_data_h_graphs(path, h_extra_features, num_classes, id_split=p.ID_SPLIT))

    logging.info('Data and H-data has been loaded')


    history = []
    history.append(defaultdict(list))

    all_preds = {}
    # save_preds key order: ['preds', 'lbls', 'pid', 'cid']
    if p.PRED_PATH:
        for name in ['train', 'val', 'test']:
            df = pd.read_csv(os.path.join(p.PRED_PATH, f'predictions_{name}_seed_{SEED_i}_split_{fold_num}.csv'))
            preds = df[[f'preds_{i}' for i in range(extra_features)]].values.reshape((1,-1,extra_features))
            lbls = df[[f'lbls_{i}' for i in range(extra_features)]].values.reshape((1,-1,extra_features))

            pid = df[['pid' for i in range(extra_features)]].values.reshape((1,-1,extra_features))
            cid = df[['cid' for i in range(extra_features)]].values.reshape((1,-1,extra_features))

            all_preds[name] = torch.tensor(np.concatenate((preds, lbls, pid, cid), 0))
        print(f'Predictions loaded from {p.PRED_PATH}',flush=True)

    else:
        # load model
        model_file=p.MODEL_FILE.format(SEED_i, fold_num)
        model = torch.load(model_file, map_location=torch.device(device), weights_only=False)
        logging.info(f'{model_file} of type {type(model)} model has been loaded')

        if isinstance(model, dict):
            # print('Dict!: ', model)
            model = model['model']

        loss_fn = torch.nn.CrossEntropyLoss().to(device) #torch.nn.BCELoss()
        logging.info(f'{type(loss_fn)} is used as loss function')

        # test
        print('Computing subgraph predictions for fold: ', fold_num, flush=True)
        model.eval()
        all_preds = {}
        for name, loader in zip(['test', 'val', 'train'], [test_loader, val_loader, train_loader]):
            print('Predicting: ', name)
            (preds, keys), res = extract_features(model,
                                       loader,
                                       metrics=metrics,
                                       print_output=True,
                                       number_classes=extra_features,
                                       device=device,
                                       no_metric=True,
                                       final_layer=p.LAYER_OF_SUB)

            all_preds[name] = preds
            # save preds
            save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_{name}_seed_{SEED_i}_split_{fold_num}.csv'), cls_num=extra_features)
        print('Predictions computed',flush=True)

    if p.ONLY_ENCODE:
        continue

    if p.NORMALIZE_SUBGRAPH_PREDS:
        train_mean = all_preds['train'][0].mean(0, keepdim=True)
        train_std  = all_preds['train'][0].std(0, keepdim=True)

        all_preds['train'][0] = (all_preds['train'][0] - train_mean) / (train_std + 1e-8)
        all_preds['val'][0] = (all_preds['val'][0] - train_mean) / (train_std + 1e-8)
        all_preds['test'][0] = (all_preds['test'][0] - train_mean) / (train_std + 1e-8)

    # train H model
    # load model
    num_features = extra_features + 1 if p.ADD_STAGE else extra_features
    from utils.model import GNN
    match p.MODEL_NAME.lower():
        case 'gine3':
            modelh = GNN(num_features, device=device, **p.KWARGS).to(device)
        case _:
            raise Exception(f'{p.MODEL_NAME} is not a valid modelh name')
    logging.info(f'{type(modelh)} model has been loaded')

    # train
    dsh_all = {'train':dsh_train, 'val':dsh_val, 'test':dsh_test}
    for ds_name in all_preds:
        curr_preds = all_preds[ds_name]
        curr_dsh = dsh_all[ds_name]
        # keys: ['preds', 'lbls', 'pid', 'cid']
        pred_dict = {str(int(curr_preds[3][i][0])):curr_preds[0][i] for i in range(len(curr_preds[0]))}
        for j in range(len(curr_dsh)):
            if not curr_dsh[j] == None and len(curr_dsh[j].nodes.shape):
                for i in range(len(curr_dsh[j].nodes)):
                    cid = curr_dsh[j].nodes[i].split('__')[5]
                    if p.ADD_STAGE:
                        curr_dsh[j].x[i][1:] = pred_dict.pop(cid)
                        pid = int(curr_dsh[j].nodes[i].split('__')[1])
                        if pid in df_l.index:
                            curr_dsh[j].x[i][0] = 0.0 if df_l.at[pid, p.STAGE_COLUMN] in p.STAGE_CLASS_0 else 1.0
                        else:
                            print(f'Could not find {pid} in clinical data')
                            curr_dsh[j].x[i][0] = 0.5
                    else:
                        curr_dsh[j].x[i][:] = pred_dict.pop(cid)

            else:
                curr_dsh[j] = None
        curr_dsh_ = []
        for l in curr_dsh:
            if not l == None:
                curr_dsh_.append(l)
        dsh_all[ds_name] = curr_dsh_

    test_h_loader = DataLoader(dsh_all['test'], batch_size=p.BS, shuffle=False)
    train_h_loader = DataLoader(dsh_all['train'], batch_size=p.BS, num_workers=p.NUM_WORKERS, shuffle=True,)
    val_h_loader = DataLoader(dsh_all['val'], batch_size=p.BS, num_workers=p.NUM_WORKERS, shuffle=True)
    train_val_h_loader = DataLoader(dsh_all['train'] + dsh_all['val'], batch_size=p.BS, num_workers=p.NUM_WORKERS, shuffle=True)

    if p.USE_LOSS_WEIGHTS:
        loss_weight = sum([i.y for i in dsh_all['train']])[0]
        loss_weight = [sum(loss_weight)/(len(loss_weight)*i) for i in loss_weight]
        print('Using loss weights: ',loss_weight)
        loss_weight = torch.tensor(loss_weight,device=device)
    else:
        loss_weight = None

    # optimizer
    match p.OPTIMIZER_NAME.lower():
        case 'adam': # default
            optimizer = torch.optim.Adam(modelh.parameters(), lr=float(p.LR))
        case 'adamw': # default
            optimizer = torch.optim.AdamW(modelh.parameters(), lr=float(p.LR), weight_decay=p.OPTARGS.get("weight_decay", 0.01))
        case 'sgd':
            optimizer = torch.optim.SGD(modelh.parameters(), lr=float(p.LR), momentum=p.OPTARGS.get("momentum", 0.0))
        case _:
            raise ValueError(f"Unknown OPTIMIZER_NAME: '{p.OPTIMIZER_NAME}'. Choose from 'adam', 'adamw', 'sgd'.")
    logging.info(f'{type(optimizer)} is used as optimizer')

    # scheduler
    match p.SCHEDULER_NAME.lower():
        case 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=p.SCHEDARGS.get("gamma", 0.9))
        case 'lin':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        case 'none': # default
            scheduler = None
        case _:
            raise ValueError(f"Unknown SCHEDULER_NAME: '{p.SCHEDULER_NAME}'. Choose from 'exp', 'lin', 'none'.")
    logging.info(f'{type(scheduler)} is used as scheduler')

    # loss
    match p.LOSS_NAME.lower():
        case 'cel': # default
            loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight).to(device)
        case 'bce':
            loss_fn = torch.nn.BCELoss(weight=loss_weight).to(device)
        case _:
            raise ValueError(f"Unknown LOSS_NAME: '{p.LOSS_NAME}'. Choose from 'cel', 'bce'.")
    logging.info(f'{type(loss_fn)} is used as loss function')



    for epoch in range(p.EPOCHS):
        average_precision_score.reset()
        auroc.reset()

        print(f'\nEPOCH {epoch}, LR {(epoch>0 and scheduler and scheduler.get_last_lr()[0] or float(p.LR)):.6f}: ', end='')
        for i,g in enumerate(optimizer.param_groups):
            print(f'LR{i}: {g["lr"]:.2e}  ',end='')
        print()
        history[-1]['epoch'].append(epoch)

        # Make sure gradient tracking is on, and do a pass over the data
        modelh.train(True)
        modelh, avg_loss, metrics_output = train_one_epoch(
            train_h_loader,
            modelh,
            optimizer,
            loss_fn,
            epoch, metrics=metrics,
            device=device,
            augmentation=augmentation)

        if not scheduler == None:
            scheduler.step()


        history[-1]['train_loss'].append(avg_loss)

        print(f'train loss: {avg_loss:<20.5} {metrics_output["Acc"]:<10.5}', end='\r')
        for resname in metrics_output:
            if 'train_'+resname in history[-1]:
                history[-1]['train_'+resname].append(metrics_output[resname])
            else:
                history[-1]['train_'+resname] = [metrics_output[resname]]


        if not (epoch + 1) % p.VAL_EPOCH:
            modelh.eval()
            print('Validation perf: ',end='')
            (preds, keys), res = epoch_validation(modelh,
                                      val_h_loader,
                                      loss_fn,
                                      print_output=True,
                                      metrics=metrics,
                                      number_classes=num_classes,
                                      device=device)

            # save progress
            history[-1]['val_x'].append(epoch)
            for resname in res:
                if 'val_'+resname in history[-1]:
                    history[-1]['val_'+resname].append(res[resname])
                else:
                    history[-1]['val_'+resname] = [res[resname]]
            if epoch > 2:
                history[-1][p.SAVE_BEST_METRIC+'smth'].append(my_mean(history[-1][p.SAVE_BEST_METRIC][-3:]))
                if max(history[-1][p.SAVE_BEST_METRIC+'smth']) == history[-1][p.SAVE_BEST_METRIC+'smth'][-1]:
                    # save best model
                    torch.save(modelh, os.path.join(p.results_path, os.path.join('best', f'{p.MODEL_NAME}_h_best_split_{fold_num}.pt')))
                    checkpoint = { # optionally just store state_dict() of each
                        'model': modelh,
                        'optimizer': optimizer,
                        'scheduler': scheduler
                    }
                    torch.save(checkpoint, os.path.join(p.results_path, f'best/checkpoint_{p.MODEL_NAME}_h_best_split_{fold_num}.pt'))
                    with open(os.path.join(p.results_path, os.path.join('best', f'{p.MODEL_NAME}_h_best_split_{fold_num}.yaml')), 'w') as f:
                        yaml.safe_dump({i:float(history[-1][i][-1]) for i in history[-1]}, f)

    # save last validation results
    save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_h_val_split_{fold_num}.csv'), cls_num=num_classes)

    # save model
    torch.save(modelh, os.path.join(p.results_path, f'{p.MODEL_NAME}_h_split_{fold_num}.pt'))

    # test
    print(f'Test Results Last (epoch={epoch-1})')
    modelh.eval()
    (preds, keys), res = evaluate(modelh, test_h_loader, metrics=metrics,
                               print_output=True, number_classes=num_classes, device=device)
    # save preds
    save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_h_test_split_{fold_num}.csv'), cls_num=num_classes)
    for resname in res:
        history[-1]['test_'+resname] = res[resname]


    print(f'Train Results Last (epoch={epoch})')
    (preds, keys), res = evaluate(modelh, train_h_loader, metrics=metrics,
                               print_output=True, number_classes=num_classes, device=device)
    # save preds
    save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_h_train_split_{fold_num}.csv'), cls_num=num_classes)


    # flush model out from GPU
    del modelh
    gc.collect()
    torch.cuda.empty_cache()

    with open(os.path.join(p.results_path, os.path.join('best', f'{p.MODEL_NAME}_h_best_split_{fold_num}.yaml')), 'r') as f:
        best_info = SimpleNamespace(**yaml.safe_load(f))
    best_epoch = best_info.epoch

    # load model

    checkpoint = torch.load(os.path.join(p.results_path, os.path.join('best', f'checkpoint_{p.MODEL_NAME}_h_best_split_{fold_num}.pt')), map_location=device, weights_only=False)
    modelhbest = checkpoint['model']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    modelhbest.device = device
    logging.info(f'{type(modelhbest)} model (epoch {best_epoch}) has been reloaded')
    modelhbest.eval()

    print(f'Test Results Best (epoch={best_epoch})')
    (preds, keys), res = evaluate(modelhbest, test_h_loader, metrics=metrics,
                               print_output=True, number_classes=num_classes, device=device)
    # save best test preds
    save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_hbest_test_split_{fold_num}.csv'), cls_num=num_classes)
    for resname in res:
        history[-1]['test_best_'+resname] = res[resname]

    print(f'Val Results Best (epoch={best_epoch})')
    (preds, keys), res = evaluate(modelhbest, val_h_loader, metrics=metrics,
                               print_output=True, number_classes=num_classes, device=device)
    # save best val preds
    save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_hbest_val_split_{fold_num}.csv'), cls_num=num_classes)
    for resname in res:
        history[-1]['val_best_'+resname] = res[resname]

    print(f'Train Results Best (epoch={best_epoch})')
    (preds, keys), res = evaluate(modelhbest, train_h_loader, metrics=metrics,
                               print_output=True, number_classes=num_classes, device=device)
    # save best val preds
    save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_hbest_train_split_{fold_num}.csv'), cls_num=num_classes)
    for resname in res:
        history[-1]['train_best_'+resname] = res[resname]



    if p.FINAL_EPOCHS_INCLUDING_VALIDATION_SET > 0:
        print(f'\nFinal training over {p.FINAL_EPOCHS_INCLUDING_VALIDATION_SET} epochs combining test+validation data')
        for epoch in range(p.FINAL_EPOCHS_INCLUDING_VALIDATION_SET):
            average_precision_score.reset()
            auroc.reset()

            print(f'\nEPOCH {best_epoch + epoch + 1}, LR {(epoch>0 and scheduler and scheduler.get_last_lr()[0] or float(p.LR)):.6f}: ', end='')
            for i,g in enumerate(optimizer.param_groups):
                print(f'LR{i}: {g["lr"]:.2e}  ',end='')
            print()
            history[-1]['epoch'].append(epoch)

            # Make sure gradient tracking is on, and do a pass over the data
            modelhbest.train(True)
            modelhbest, avg_loss, metrics_output = train_one_epoch(
                train_val_h_loader,
                modelhbest,
                optimizer,
                loss_fn,
                best_epoch + epoch + 1, metrics=metrics,
                device=device,
                augmentation=augmentation)

            if not scheduler == None:
                scheduler.step()


            history[-1]['train_loss'].append(avg_loss)

            print(f'train loss: {avg_loss:<20.5} {metrics_output["Acc"]:<10.5}', end='\r')
            for resname in metrics_output:
                if 'train_'+resname in history[-1]:
                    history[-1]['train_'+resname].append(metrics_output[resname])
                else:
                    history[-1]['train_'+resname] = [metrics_output[resname]]

        # save (presumed) improved model
        torch.save(modelhbest, os.path.join(p.results_path, f'best/{p.MODEL_NAME}_h_best_split_{fold_num}.pt'))

        modelhbest.eval()

        print(f'Test Results Best (epoch={best_epoch} + {epoch+1})')
        (preds, keys), res = evaluate(modelhbest, test_h_loader, metrics=metrics,
                                print_output=True, number_classes=num_classes, device=device)
        # save best test preds
        save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_hbest_test_split_{fold_num}.csv'), cls_num=num_classes)
        for resname in res:
            history[-1]['test_best_'+resname] = res[resname]

        print(f'Val Results Best (epoch={best_epoch} + {epoch+1})')
        (preds, keys), res = evaluate(modelhbest, val_h_loader, metrics=metrics,
                                print_output=True, number_classes=num_classes, device=device)
        # save best val preds
        save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_hbest_val_split_{fold_num}.csv'), cls_num=num_classes)
        for resname in res:
            history[-1]['val_best_'+resname] = res[resname]

        print(f'Train Results Best (epoch={best_epoch} + {epoch+1})')
        (preds, keys), res = evaluate(modelhbest, train_h_loader, metrics=metrics,
                                print_output=True, number_classes=num_classes, device=device)
        # save best train preds
        save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_hbest_train_split_{fold_num}.csv'), cls_num=num_classes)
        for resname in res:
            history[-1]['train_best_'+resname] = res[resname]


    # save history
    with open(os.path.join(p.results_path, f'history_h_{fold_num}.csv'), 'w') as f:
        fieldnames = list(dict.fromkeys(k for hist in history for k in hist))
        w = csv.DictWriter(f, fieldnames, restval='')
        w.writeheader()
        for hist in history:
            w.writerow(hist)
