# -*- coding: utf-8 -*-
"""
Training utilities: data loading, metrics, and train/eval loops.
"""

import torch
import csv
import logging
import numpy as np
import pickle

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data


class FeatureNoise():
    """Augmentation that adds Gaussian noise to node features.

    Args:
        sigma: Standard deviation of the noise.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        data.x = torch.add(data.x, torch.randn(data.x.shape), alpha=self.sigma)
        return data


def load_data_subgraphs(path, num_classes, mode='cls'):
    """Load subgraph .dat files into a list of PyG Data objects.

    Args:
        path: Path to the pickled subgraph data file.
        num_classes: Number of output classes (used to build one-hot labels).
        mode: Label encoding mode; only 'cls' (classification) is supported.

    Returns:
        List of PyG Data objects with node features, edges, labels, and IDs.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    adj, feat, lbl, ids, f_ids = data['Adj'], data['Features'], data['Labels'], data['Id'], data['FullID']
    length = len(feat)
    ds_ = []
    for i in range(length):
        res = from_scipy_sparse_matrix(adj[i])
        edges = res[0].type(torch.int64)
        weights = res[1].type(torch.float)
        feat_tmp = torch.tensor(feat[i], dtype=torch.float)
        match mode.lower():
            case 'cls':
                lbl_one_hot = [0 for j in range(num_classes)]
                lbl_one_hot[lbl] = 1
                lbl_tmp = torch.tensor([lbl_one_hot], dtype=torch.float)
            case _:
                raise ValueError(f'{mode} is not a valid mode')
        id_tmp = torch.tensor([[ids]], dtype=torch.float)
        fid_tmp = torch.tensor(int(f_ids[i].split('__')[5]), dtype=torch.int64)
        ds_.append(Data(x=feat_tmp, edge_index=edges, edge_attr=weights, y=lbl_tmp, pid=id_tmp, cluster_id=fid_tmp))
    return ds_


def load_data_h_graphs(path, num_features, num_classes, mode='cls', id_split='Lung # '):
    """Load hierarchical core-graph .dat files into a list of PyG Data objects.

    Args:
        path: Path to the pickled hierarchical graph data file.
        num_features: Number of node features per subgraph node.
        num_classes: Number of output classes (used to build one-hot labels).
        mode: Label encoding mode; only 'cls' (classification) is supported.
        id_split: Delimiter used to parse the patient ID from the graph ID string.

    Returns:
        List of PyG Data objects with node features, edges, labels, and IDs.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    adj, feat, lbl, ids, nodes = data['Adj'], data['Features'], data['Labels'], data['IDs'], data['Nodes']
    length = len(feat)
    ds_ = []
    id_list = []
    for i in range(length):
        try:
            res = from_scipy_sparse_matrix(adj[i])
        except Exception:
            print('### adj: ', i)
            raise
        edges = res[0].type(torch.int64)
        weights = res[1].type(torch.float)

        try:
            feat_tmp = torch.tensor([[0 for k in range(num_features)] for j in feat[i]], dtype=torch.float)
        except Exception:
            print('X### ids: ', ids)
            print('X### feat: ', i, feat)
            raise
        match mode.lower():
            case 'cls':
                lbl_one_hot = [0 for j in range(num_classes)]
                lbl_one_hot[int(lbl[i])] = 1
                lbl_tmp = torch.tensor([lbl_one_hot], dtype=torch.float)
            case _:
                raise ValueError(f'{mode} is not a valid mode')

        id_tmp = int(ids[i].split(id_split)[0])*10
        while id_tmp in id_list:
            id_tmp += 1
        id_tmp = torch.tensor(id_tmp, dtype=torch.int64)
        nodes_tmp = np.squeeze(nodes[i])
        ds_.append(Data(x=feat_tmp, edge_index=edges, edge_attr=weights, y=lbl_tmp, pid=id_tmp, nodes=nodes_tmp))
    return ds_


def my_mean(a):
    return sum(a)/len(a)


def save_preds(preds, keys, save_path='res.csv', cls_num=2):
    """Save model predictions and labels to a CSV file.

    Args:
        preds: Tensor of shape (4, N, num_classes) with predictions, labels, pid, cid.
        keys: List of string keys corresponding to the first axis of preds.
        save_path: Output CSV file path.
        cls_num: Number of classes, used to build CSV column names.
    """
    with open(save_path, 'w') as f:
        fieldnames = [f'preds_{i}' for i in range(cls_num)] + [f'lbls_{i}' for i in range(cls_num)] + ['pid', 'cid']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(preds[0])):
            d_tmp = {}
            for k_index, k in enumerate(keys):
                if k in ['preds', 'lbls']:
                    for ii in range(len(preds[0, i])):
                        try:
                            d_tmp[k+f'_{ii}'] = preds[k_index, i, ii].numpy()
                        except Exception:
                            d_tmp[k+f'_{ii}'] = preds[k_index, i, ii].detach().cpu().numpy()
                else:
                    try:
                        d_tmp[k] = int(preds[k_index, i, 0].numpy())
                    except Exception:
                        d_tmp[k] = int(preds[k_index, i, 0].detach().cpu().numpy())
            writer.writerow(d_tmp)


def check_duplicate_ids(loaders):
    """Return a list of patient IDs that appear in more than one data loader.

    Args:
        loaders: List of PyG DataLoaders to check for overlapping patient IDs.

    Returns:
        List of duplicate IDs, or an empty list if no duplicates are found.
    """
    def _check_duplicate_ids(id_lists):
        duplicates_list = []
        if len(id_lists) < 2:
            return duplicates_list
        for id_tmp in id_lists[0]:
            for id_list in id_lists[1:]:
                if id_tmp in id_list:
                    duplicates_list.append(id_tmp)
        return duplicates_list + _check_duplicate_ids(id_lists[1:])

    id_lists = []
    for loader in loaders:
        tmp_id = []
        for i in loader:
            tmp_id.extend(i.pid.numpy().flatten().tolist())
        id_lists.append(list(set(tmp_id)))
    return _check_duplicate_ids(id_lists)


def acc_fn(pred, lbl):
    return torch.sum((pred>0.5) == (lbl>0.5)) / len(lbl)


def _assemble_preds(outputs, data, device='cpu'):
    """Pack model outputs and batch data into a (4, batch_size, num_classes) tensor.

    Row order: predictions, labels, patient IDs, cluster IDs.
    """
    current_bs = len(data)
    number_classes = outputs.shape[-1]
    res = torch.zeros((4, current_bs, number_classes), device=device)
    res[0, :, :] = outputs
    if number_classes > 1:
        res[1, :, :data.y.shape[1]] = data.y
    else:
        res[1, :, 0] = data.y
    try:
        res[2, :, 0] = data.pid[:, 0]
        res[3, :, 0] = data.cluster_id
    except Exception:
        res[2, :, 0] = data.pid
    return res


def get_preds(model, loader, number_classes=2, device='cpu'):
    """Run inference over a data loader and return stacked predictions.

    Returns:
        Tuple of (preds tensor of shape (4, N, num_classes), key list).
    """
    with torch.no_grad():
        preds = torch.zeros((4, 0, number_classes), device=device)
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.cat((preds, _assemble_preds(outputs, data, device=device)), 1)
    return preds, ['preds', 'lbls', 'pid', 'cid']


def print_metrics_output(metrics, res):
    for i, metric_name in enumerate(metrics):
        print(f'{", " if i>0 else ""}{metric_name}: {res[metric_name]:.4f}', end='')
    print()


def compute_metrics(metrics, preds):
    res = {}
    for metric_name in metrics:
        res[metric_name] = metrics[metric_name](preds[0, :], preds[1, :]).detach().cpu().numpy()
    return res


def evaluate(model, loader, metrics=None, print_output=True, number_classes=2, device='cpu', no_metric=False):
    """Evaluate a model on a data loader and optionally print metrics.

    Args:
        model: Trained PyG model.
        loader: DataLoader to evaluate on.
        metrics: Dict mapping metric names to callable metric functions.
        print_output: If True, print metric values to stdout.
        number_classes: Number of output classes.
        device: Torch device string.
        no_metric: If True, skip metric computation.

    Returns:
        Tuple of ((preds, keys), results dict).
    """
    if metrics is None:
        metrics = {}
    preds, keys = get_preds(model, loader, number_classes=number_classes, device=device)
    if not no_metric:
        res = compute_metrics(metrics, preds)
        if print_output:
            print_metrics_output(metrics, res)
    else:
        res = {}
        print('NO metrics')
    return (preds, keys), res


def get_features(model, loader, number_classes=2, device='cpu', final_layer=0):
    """Run inference and optionally extract intermediate layer activations.

    Args:
        model: Trained PyG model.
        loader: DataLoader to run inference on.
        number_classes: Number of output classes.
        device: Torch device string.
        final_layer: 0 for output layer, -1 for first hidden FC layer, -2 for lin_in layer.

    Returns:
        Tuple of (preds tensor of shape (4, N, num_classes), key list).
    """
    def hook_fn(module, input, output):
        global penultimate_output
        penultimate_output = output

    # Register hook on the requested intermediate layer
    if final_layer == -1:
        hook = model.hlin_list[0].register_forward_hook(hook_fn)
    elif final_layer == -2:
        hook = model.lin_in.register_forward_hook(hook_fn)

    with torch.no_grad():
        preds = torch.zeros((4, 0, number_classes), device=device)
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            if final_layer == 0:
                preds = torch.cat((preds, _assemble_preds(outputs, data, device=device)), 1)
            else:
                preds = torch.cat((preds, _assemble_preds(penultimate_output, data, device=device)), 1)
    if not final_layer == 0:
        hook.remove()
    return preds, ['preds', 'lbls', 'pid', 'cid']


def extract_features(model, loader, metrics=None, print_output=True, number_classes=2,
                     device='cpu', no_metric=False, final_layer=0):
    preds, keys = get_features(model, loader, number_classes=number_classes, device=device, final_layer=final_layer)
    return (preds, keys), {}


def epoch_validation(model, loader, loss_fn, print_output=True, metrics=None, number_classes=2, device='cpu'):
    """Run one validation epoch and return metrics including loss.

    Args:
        model: Trained PyG model.
        loader: Validation DataLoader.
        loss_fn: Loss function callable.
        print_output: If True, print metric values to stdout.
        metrics: Dict mapping metric names to callable metric functions.
        number_classes: Number of output classes.
        device: Torch device string.

    Returns:
        Tuple of ((preds, keys), results dict) including 'Loss'.
    """
    if metrics is None:
        metrics = {}
    metrics['Loss'] = loss_fn
    return evaluate(model, loader, metrics, print_output, number_classes=number_classes, device=device)


def train_one_epoch(training_loader, model, optimizer, loss_fn, epoch_index, metrics=None,
                    device='cpu', augmentation=None):
    """Train the model for one epoch and return loss and metrics.

    Args:
        training_loader: Training DataLoader.
        model: PyG model to train.
        optimizer: Torch optimizer.
        loss_fn: Loss function callable.
        epoch_index: Current epoch index (unused internally, available for logging by callers).
        metrics: Dict mapping metric names to callable metric functions.
        device: Torch device string.
        augmentation: Optional augmentation callable applied per batch.

    Returns:
        Tuple of (model, mean_loss, metrics_dict).
    """
    if metrics is None:
        metrics = {}
    running_loss = 0.
    metrics_output = {metric_name: 0. for metric_name in metrics}
    num_done, num_skipped = 0, 0

    for data in training_loader:
        if augmentation:
            data = augmentation(data)

        try:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, data.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            for metric_name in metrics:
                metrics_output[metric_name] += metrics[metric_name](outputs, data.y).detach().cpu().numpy()
            num_done += 1

        except Exception:
            # A batch can fail for benign reasons (e.g. pooling leaving a single
            # node, which BatchNorm rejects), so training continues -- but the
            # cause is logged rather than swallowed, since the same handler would
            # otherwise hide genuine bugs, OOM, and NaNs.
            num_skipped += 1
            logging.exception('Batch skipped during training')

    if num_skipped:
        logging.warning(f'{num_skipped} of {num_skipped + num_done} batches were skipped this epoch')
    if not num_done:
        raise RuntimeError('No batch completed successfully this epoch (see logged exceptions above)')

    # averaged over completed batches only, so skipped ones do not dilute the mean
    return model, running_loss / num_done, {metric_name: res/num_done for metric_name, res in metrics_output.items()}
