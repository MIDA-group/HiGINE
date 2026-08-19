#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train model to predict survival based on subgraphs, with cell level features.
"""

import os
import sys
import logging
import yaml
from types import SimpleNamespace
from shutil import copyfile
import pandas as pd
import numpy as np
from collections import defaultdict
import csv
import argparse
import gc
import random #Only to set seed
import utils.batchnorm_pop_stats as BNstats


logging.basicConfig(level=logging.INFO)

# set parameters
params_file = './settings/mif/train_subgraphs.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--parameters-file', default=params_file, type=str, help='file where the parameters for the training are stored')
parser.add_argument('--stdin-overrides', action='store_true', default=False, help='override parameter settings from stdin')
parser.add_argument('--results-path', default=None, type=str, help='override parameter file and stdin setting')
parser.add_argument('--no-train', action='store_true', default=False, help='Inference only, no training')
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

if args.results_path:
    p.results_path = args.results_path

if args.no_train:
    p.EPOCHS = 0
    print('Setting EPOCHS=0 due to --no-train')


# Must be set before any import torch
if hasattr(p, 'CUDA_DEVICE'):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p.CUDA_DEVICE)
# Determinisitic CUDA (https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# CUDA related imports
import torch
import torchmetrics
from torch_geometric.loader import DataLoader
from utils.model import GNN
from utils.training_utils import train_one_epoch, epoch_validation, evaluate, \
    check_duplicate_ids, save_preds, my_mean, load_data_subgraphs, FeatureNoise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))


# preprocessing parameters used
graphs_input_folder = p.graphs_path
with open(os.path.join(graphs_input_folder, p.GRAPH_PARAMS), 'r') as f:
    pre_p = SimpleNamespace(**yaml.safe_load(f))

# TODO: set the num_features in some other way
num_features = len(pre_p.FEATURE_LIST)
num_classes = 2
del pre_p # not used elsewhere


# To train or not to train
if args.no_train:
    if not os.path.exists(os.path.join(p.results_path, 'best')):
        logging.error(f'Required model path missing (with --no-train option): {p.results_path}')
        raise Exception()
    else:
        logging.info(f'Existing path used to read model (with --no-train option): {p.results_path}')
else:
    if not os.path.exists(p.results_path):
        os.makedirs(os.path.join(p.results_path, 'best')) # will also make results_path
        logging.info(f'New path for results was created: {p.results_path}')
    else:
        logging.error(f'Results path exists already: {p.results_path}')
        raise Exception(f'Folder: {p.results_path} exists already')


# copy parameters into folder
copyfile(os.path.join(graphs_input_folder, p.GRAPH_PARAMS), os.path.join(p.results_path, 'graph_PARAMS.yaml'))
copyfile(params_file, os.path.join(p.results_path, 'train_PARAMS.yaml')) # this one is loaded by train_coregraphs
copyfile(os.path.join(graphs_input_folder, p.SUBSAMPLING_PARAMS), os.path.join(p.results_path, 'subsample_PARAMS.yaml'))


# TODO: Move into utils
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
        logging.info(f'{augmentation} as augmentation used')
    case _:
        augmentation = None
        logging.info('No augmentation used')


# Compute global BN statistics instead of running average
# If this is too slow, then adjust momentum in BN layers instead
def compute_global_BNstats(model,device):
    model.apply(BNstats.adjust_bn_layers_to_compute_population_stats)
    for i in range(1): # Repeat more than once for more stable results
        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)
                _ = model(data)
    model.apply(BNstats.restore_original_settings_of_bn_layers)




# read label info
df_l = pd.read_csv(p.labels_file, index_col='ID')

# for each seed
for i_seed, seed in enumerate(p.SEED):
    history = []

    for fold_num in range(p.FOLDS):
        print(f'\nFOLD {fold_num} with SEED {seed}') #Making each fold stand-alone by resetting the seed
        history.append(defaultdict(list))

        # For reproducible training (at least if staying on the same hardware)
        # https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(seed)     # python random generator
        np.random.seed(seed)  # numpy random generator
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True) # The cumsum is possibly only in eval

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        torch_train_gen = torch.Generator()
        torch_train_gen.manual_seed(seed)
        torch_val_gen = torch.Generator()
        torch_val_gen.manual_seed(seed)

        # load data
        #train,val,test - order matters
        ds = {}
        for x in ['train','val','test']:
            ds_name = p.folds_format.format(fold_num,x)
            with open(os.path.join(p.folds_path, f'{ds_name}.csv'), 'r') as f:
                print('Reading: ',os.path.join(p.folds_path, f'{ds_name}.csv'), end='')
                reader = csv.DictReader(f)
                ids = [row['ID'] for row in reader]
                print(f'\t {len(ids):3} samples', end='')

            ds_ = []
            lbl_count = np.zeros(num_classes)
            for id_ in ids:
                path = os.path.join(graphs_input_folder,  os.path.join('files', f'{id_}_Adj_Features.dat'))
                data=load_data_subgraphs(path, num_classes)

                #pick label from csv file
                lab=df_l.loc[int(id_),'label']
                lbl_one_hot = [0 for _ in range(num_classes)]
                lbl_one_hot[lab] = 1
                lbl_count += lbl_one_hot
                for d in data:
                    d.y = torch.tensor([lbl_one_hot], dtype=torch.float)

                ds_.extend(data)
            ds[x] = ds_
            print(f'\t labels: {lbl_count}')

        logging.info('Data has been loaded')

        #Don't drop_last if you have very few datapoints
        train_loader = DataLoader(ds['train'], batch_size=p.BS, num_workers=p.NUM_WORKERS, worker_init_fn=seed_worker, generator=torch_train_gen, shuffle=True, drop_last=True, pin_memory=True)
        val_loader = DataLoader(ds['val'], batch_size=p.BS, num_workers=p.NUM_WORKERS, worker_init_fn=seed_worker, generator=torch_val_gen, shuffle=True, pin_memory=True)
        test_loader = DataLoader(ds['test'], batch_size=p.BS, num_workers=p.NUM_WORKERS, shuffle=False, pin_memory=True)
        logging.info(f'DataLoaders have been created: {len(train_loader.dataset)},{len(val_loader.dataset)},{len(test_loader.dataset)}')

        if p.FINAL_EPOCHS_INCLUDING_VALIDATION_SET > 0:
            train_val_loader = DataLoader(torch.utils.data.ConcatDataset([ds['train'],ds['val']]), batch_size=p.BS, num_workers=p.NUM_WORKERS, worker_init_fn=seed_worker, generator=torch_train_gen, shuffle=True, drop_last=True, pin_memory=True)


        if p.USE_LOSS_WEIGHTS:
            loss_weight = [j.item() for j in sum([i.y for i in ds['train']])[0]]
            loss_weight = torch.tensor([sum(loss_weight)/(len(loss_weight)*i) for i in loss_weight]).to(device)
        else:
            loss_weight = None
        logging.info(f'Using loss weights: {loss_weight}')


        # Assert that there are no duplicates in the data (no leakage)
        dupes = check_duplicate_ids([val_loader, test_loader, train_loader])
        if len(dupes):
           logging.error(f'There are {len(dupes)} duplicates in the datasets')
        else:
           logging.info('No duplicates found')


        # load model
        model = GNN(num_features, device=device, **p.KWARGS).to(device)
        logging.info(f'{type(model)} model has been loaded')


        # optimizer
        match p.OPTIMIZER_NAME.lower():
            case 'adam': # default
                optimizer = torch.optim.Adam(model.parameters(), lr=float(p.LR), **p.OPTARGS)
            case 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=float(p.LR), **p.OPTARGS)
            case 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=float(p.LR), **p.OPTARGS)
            case _:
                raise ValueError(f"Unknown OPTIMIZER_NAME: '{p.OPTIMIZER_NAME}'. Choose from 'adam', 'adamw', 'sgd'.")
        logging.info(f'{type(optimizer)} is used as optimizer')

        # scheduler
        match p.SCHEDULER_NAME.lower():
            case 'exp':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **p.SCHEDARGS)
            case 'none': # default
                scheduler = None
            case _:
                raise ValueError(f"Unknown SCHEDULER_NAME: '{p.SCHEDULER_NAME}'. Choose from 'exp', 'none'.")
        logging.info(f'{type(scheduler)} is used as scheduler')

        # loss
        match p.LOSS_NAME.lower():
            case 'cel': # default
                loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight, **p.LOSSARGS).to(device)
            case 'bce':
                loss_fn = torch.nn.BCELoss(weight=loss_weight, **p.LOSSARGS).to(device)
            case _:
                raise ValueError(f"Unknown LOSS_NAME: '{p.LOSS_NAME}'. Choose from 'cel', 'bce'.")
        logging.info(f'{type(loss_fn)} is used as loss function')
        metrics['Loss'] = loss_fn



        # Training
        def train(model, epoch, train_loader, val_loader=None, test_loader=None):
            average_precision_score.reset()
            auroc.reset()

            print(f'\nEPOCH {epoch}, LR {(epoch>0 and scheduler and scheduler.get_last_lr()[0] or float(p.LR)):.6f}: ', end='')
            for i,g in enumerate(optimizer.param_groups):
                print(f'LR{i}: {g["lr"]:.2e}  ',end='')
            print()
            history[-1]['epoch'].append(epoch)

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            model, avg_loss, metrics_output = train_one_epoch(train_loader, model, optimizer, loss_fn,
                                                            epoch,
                                                            metrics=metrics,
                                                            device=device,
                                                            augmentation=augmentation)

            history[-1]['train_loss'].append(avg_loss)
            for resname in metrics_output:
                history[-1]['train_'+resname].append(metrics_output[resname])


            # Evaluation part (only for logging)
            model.eval()
            compute_global_BNstats(model,device)

            print('Train set perf:  ', end='') # Differs slightly from running train set perf. since now at end of epoch
            (preds, keys), res = epoch_validation(model, train_loader, loss_fn, print_output=True,
                                                    metrics=metrics,
                                                    number_classes=num_classes,
                                                    device=device)

            if val_loader:
                print('Validation perf: ', end='')
                (preds, keys), res = epoch_validation(model, val_loader, loss_fn, print_output=True,
                                                    metrics=metrics,
                                                    number_classes=num_classes,
                                                    device=device)
                # save progress
                for resname in res:
                    history[-1]['val_'+resname].append(res[resname])

            if test_loader:
                print('Test set perf:   ', end='')
                (preds, keys), res = epoch_validation(model, test_loader, loss_fn, print_output=True,
                                                    metrics=metrics,
                                                    number_classes=num_classes,
                                                    device=device)
                # save progress
                for resname in res:
                    history[-1]['test_'+resname].append(res[resname])


            if not scheduler == None:
                scheduler.step() #metrics=metrics['Loss'])

            if epoch > p.MIN_EPOCHS and (p.SAVE_BEST_METRIC in history[-1]) and val_loader:
                history[-1][p.SAVE_BEST_METRIC+'smth'].append(my_mean(history[-1][p.SAVE_BEST_METRIC][-2:])) #Average of last two epochs
                if max(history[-1][p.SAVE_BEST_METRIC+'smth']) == history[-1][p.SAVE_BEST_METRIC+'smth'][-1]:
                    # save best model
                    checkpoint = { # optionally just store state_dict() of each
                        'model': model,
                        'optimizer': optimizer,
                        'scheduler': scheduler
                    }
                    torch.save(checkpoint, os.path.join(p.results_path, f'best/{p.MODEL_NAME}_best_seed_{seed}_split_{fold_num}.pt'))
                    # save yaml history for best
                    with open(os.path.join(p.results_path, f'best/{p.MODEL_NAME}_best_seed_{seed}_split_{fold_num}.yaml'), 'w') as f:
                        yaml.safe_dump({i:float(history[-1][i][-1]) for i in history[-1]}, f)



        # train over epochs
        for epoch in range(p.EPOCHS):
            train(model, epoch, train_loader, val_loader, test_loader)

        # flush model out from GPU
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # load best model
        checkpoint = torch.load(os.path.join(p.results_path, os.path.join('best', f'{p.MODEL_NAME}_best_seed_{seed}_split_{fold_num}.pt')), map_location=device, weights_only=False)
        modelbest = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        with open(os.path.join(p.results_path, f'best/{p.MODEL_NAME}_best_seed_{seed}_split_{fold_num}.yaml'), 'r') as f:
            best_info = SimpleNamespace(**yaml.safe_load(f))
        best_epoch = best_info.epoch

        logging.info(f'{type(modelbest)} model (epoch {best_epoch}) has been reloaded')

        # confirm that we get the same result as earlier
        print(f'Test Results (epoch {best_epoch})')
        modelbest.eval()
        (preds, keys), res = evaluate(modelbest, test_loader, metrics=metrics,
                                   print_output=True, number_classes=num_classes,
                                   device=device)


        # final training including the validation set
        if p.FINAL_EPOCHS_INCLUDING_VALIDATION_SET > 0:
            print(f'\nFinal training over {p.FINAL_EPOCHS_INCLUDING_VALIDATION_SET} epochs combining test+validation data')
            for extra_epoch in range(p.FINAL_EPOCHS_INCLUDING_VALIDATION_SET):
                train(modelbest, best_epoch + extra_epoch+1, train_val_loader, None, test_loader)
                # train(modelbest, best_epoch + extra_epoch+1, val_loader, None, test_loader)

            print(f'Test Results (epoch {best_epoch} + {extra_epoch+1})')
            modelbest.eval()
            (preds, keys), res = evaluate(modelbest, test_loader, metrics=metrics,
                                    print_output=True, number_classes=num_classes,
                                    device=device)

            # save (presumed) improved model
            torch.save(modelbest, os.path.join(p.results_path, f'best/{p.MODEL_NAME}_best_seed_{seed}_split_{fold_num}.pt'))
            # save yaml history for improved best
            # TODO

        # save test best preds
        save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_best_test_seed_{seed}_split_{fold_num}.csv'), cls_num=num_classes)
        for resname in res:
            history[-1]['test_best_'+resname] = res[resname]

        print(f'Validation Results (epoch {best_epoch}) - {"just to verify" if not p.FINAL_EPOCHS_INCLUDING_VALIDATION_SET else "part of final training data"}')
        (preds, keys), res = evaluate(modelbest, val_loader, metrics=metrics,
                                   print_output=True, number_classes=num_classes,
                                   device=device)

        # save val best preds
        save_preds(preds, keys, save_path=os.path.join(p.results_path, f'predictions_best_val_seed_{seed}_split_{fold_num}.csv'), cls_num=num_classes)
        for resname in res:
            history[-1]['val_best_'+resname] = res[resname]

        # save csv history
        with open(os.path.join(p.results_path, f'history_seed_{seed}_{fold_num}.csv'), 'w') as f:
            fieldnames = list(dict.fromkeys(k for hist in history for k in hist))
            w = csv.DictWriter(f, fieldnames, restval='')
            w.writeheader()
            for hist in history:
                w.writerow(hist)

