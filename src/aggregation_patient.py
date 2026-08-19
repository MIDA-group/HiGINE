#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate core-level predictions to patient level and evaluate the result.
"""

import yaml
import os
import argparse
from types import SimpleNamespace
from utils.training_utils import my_mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, auc, roc_curve, \
    average_precision_score, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter


font = {'size'   : 14}
matplotlib.rc('font', **font)

parameters_file = './settings/mif/aggregation_patient.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--parameters-file', default=parameters_file, type=str, help='file where the parameters for the training are stored')
parser.add_argument('--core-path', default=None, type=str, help='override parameter file setting')
args = parser.parse_args()
parameters_file = args.parameters_file

# load parameters and previous parameters
with open(parameters_file, 'r') as f:
    train_para = yaml.safe_load(f)
p = SimpleNamespace(**train_para)


# how to show (or not show) plots
if p.SHOW_PLOTS:
    if p.BLOCK_ON_PLOTS:
        showplot=lambda:plt.show()
    else:
        showplot=lambda:plt.pause(0.001)
else:
    showplot=lambda:plt.close()


path_complete = os.path.join(p.base_path, p.core_path)
if args.core_path:
    path_complete = args.core_path

num_runs = p.FOLDS
num_cls = 2
class_id = 0
ds_name = 'test'


label_data = p.LABEL_DATA

df_l = pd.read_csv(label_data)

df_l.index = df_l['ID']
df_l["event"] = df_l["Dead/Alive"].map(lambda x: x == "Dead")
df_l["time"] = df_l["Follow-up (days)"]



def aggregate_pid(df, agg_meth='mean', num_cls=None):
    """
    Aggregate n-class predictions at the patient (pid) level.

    Args:
        df (pd.DataFrame): DataFrame with columns:
            - 'pid': patient ID
            - 'preds_0' ... 'preds_{num_cls-1}': predicted probabilities
            - 'lbls_0' ... 'lbls_{num_cls-1}': one-hot true labels (optional)
        agg_meth (str): aggregation method, one of
            ['mean', 'median', 'max', 'majority', 'weighted']
        num_cls (int, optional): number of classes (inferred if None)

    Returns:
        pd.DataFrame: aggregated patient-level predictions and labels
    """
    # Infer number of classes
    if num_cls is None:
        pred_cols = [c for c in df.columns if c.startswith('preds_')]
        num_cls = len(pred_cols)

    pred_cols = [f'preds_{i}' for i in range(num_cls)]
    lbl_cols = [f'lbls_{i}' for i in range(num_cls) if f'lbls_{i}' in df.columns]

    grouped = df.groupby('pid')
    pid_list, pred_list, true_list = [], [], []

    for pid, group in grouped:
        preds = group[pred_cols].values

        # --- Aggregation ---
        if agg_meth == 'mean':
            agg_pred = preds.mean(axis=0)
        elif agg_meth == 'median':
            agg_pred = np.median(preds, axis=0)
        elif agg_meth == 'max':
            agg_pred = preds.max(axis=0)
        elif agg_meth == 'majority':
            hard = np.argmax(preds, axis=1)
            agg_label = np.bincount(hard, minlength=num_cls) / len(hard)
            agg_pred = agg_label
        elif agg_meth == 'weighted':
            # Weight by confidence (inverse entropy)
            ent = -(preds * np.log(preds + 1e-8)).sum(axis=1)
            weights = 1 / (ent + 1e-8)
            weights /= weights.sum()
            agg_pred = np.average(preds, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation method: {agg_meth}")

        # --- True label aggregation (optional) ---
        if lbl_cols:
            true_vals = group[lbl_cols].values
            true_label = true_vals.mean(axis=0)  # should be identical anyway
        else:
            true_label = np.nan

        pid_list.append(pid)
        pred_list.append(agg_pred)
        true_list.append(true_label)

    pred_arr = np.vstack(pred_list)
    true_arr = np.vstack(true_list) if len(lbl_cols) > 0 else None

    out_df = pd.DataFrame(pred_arr, columns=[f'preds_{i}' for i in range(num_cls)])
    out_df.insert(0, 'pid', pid_list)

    if true_arr is not None:
        for i in range(num_cls):
            out_df[f'lbls_{i}'] = true_arr[:, i]

    return out_df

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Define aggregation methods to test
agg_methods = p.NON_LEARNING_METHODS

results_summary = {}

for agg_method in agg_methods:
    method_name = 'No_agg' if agg_method is None else agg_method
    print(f"\n{'='*60}")
    print(f"Aggregation Method: {method_name.upper()}")
    print(f"{'='*60}")

    all_roc, all_pr, all_cindex = [], [], []
    per_split = []

    for split_num in range(num_runs):
        path = os.path.join(path_complete, p.HBEST_FILE.format(ds_name, split_num))
        df = pd.read_csv(path)

        pred = df[[f'preds_{i}' for i in range(num_cls)]].values
        pred = [softmax(j) for j in pred]
        df[[f'preds_{i}' for i in range(num_cls)]] = pred

        if agg_method:
            df = aggregate_pid(df, agg_meth=agg_method, num_cls=num_cls)

        pred = df[[f'preds_{i}' for i in range(num_cls)]].values
        true = df[[f'lbls_{i}' for i in range(num_cls)]].values

        fpr, tpr, _ = roc_curve([i[1] for i in true], [i[1] for i in pred])
        all_roc.append(auc(fpr, tpr))
        all_pr.append(average_precision_score([i[1] for i in true], [i[1] for i in pred]))

        # pid holds the patient ID multiplied by 10; load_data_h_graphs (in
        # utils/training_utils.py) uses the freed-up last digit to keep several
        # cores of the same patient distinct, so //10 recovers the patient ID
        y_event_time = df.pid.map(lambda x: df_l.loc[df_l['ID'] == int(x//10), 'Follow-up (days)'].to_list()[0])
        y_event = df.pid.map(lambda x: df_l.loc[df_l['ID'] == int(x//10), 'Dead/Alive'].to_list()[0] == 'Dead')
        all_cindex.append(concordance_index_censored(y_event, y_event_time, [i[0] for i in pred])[0])

        per_split.append((true, pred))

    results_summary[method_name] = {'ROC': all_roc, 'PR': all_pr, 'C-Index': all_cindex}

    print(f'AUROC:   {my_mean(all_roc):.4f}')
    print(f'PR AUC:  {my_mean(all_pr):.4f}')
    print(f'C-Index: {my_mean(all_cindex):.4f}')

    # one figure per curve type, drawn from the predictions collected above
    for func in [PrecisionRecallDisplay, RocCurveDisplay]:
        fig, ax = plt.subplots(figsize=(5, 5))
        pcl = True
        for split_num, (true, pred) in enumerate(per_split):
            display = func.from_predictions(
                true[:, class_id],
                pred[:, class_id],
                name=f"split: {split_num}",
                ax=ax,
                plot_chance_level=pcl
            )
            pcl = False

        _ = display.ax_.set(
            title=f"Class id {class_id} - {method_name}",
        )
        showplot()

print("\n" + "="*60)
print("SUMMARY OF ALL METHODS")
print("="*60)


# Print summary table
print(f"\n{'Method':<20} {'ROC AUC':<15} {'PR AUC':<15} {'C-Index':<15}")
print("-" * 50)
for method, metrics in results_summary.items():
    print(f"{method:<20} {my_mean(metrics['ROC']):<15.4f} {my_mean(metrics['PR']):<15.4f} {my_mean(metrics['C-Index']):<15.4f}")
print("\n")

metrics = ['ROC', 'PR', 'C-Index']
methods = list(results_summary.keys())

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.suptitle("Performance Metrics Across Aggregation Methods", fontsize=14, fontweight='bold')

for i, metric in enumerate(metrics):
    data = [results_summary[m][metric] for m in methods]
    axes[i].boxplot(data)
    axes[i].set_title(metric, fontsize=12)
    axes[i].set_xticks(range(1, len(methods) + 1))
    axes[i].set_xticklabels(methods, rotation=30, ha='right')
    axes[i].set_ylabel("Score")
    axes[i].grid(alpha=0.7)
    axes[i].plot([0.5, 6.5], [0.5, 0.5], 'grey', linestyle='--')
    axes[i].set_ylim((0, 1))
plt.tight_layout(rect=[0, 0, 1, 0.95])
showplot()


def make_patient_features(df, num_cls, use_stage=False):
    """
    Convert per-sample predictions into per-patient feature vectors.

    Args:
        df (pd.DataFrame): DataFrame with per-sample predictions.
            Must contain:
                - 'pid' column (patient ID)
                - 'preds_0' ... 'preds_{num_cls-1}'
                - 'lbls_0' ... 'lbls_{num_cls-1}'
                - Optional: 'stage' (patient-level metadata)
        num_cls (int): number of classes
        use_stage (bool): whether to include the 'stage' column in features

    Returns:
        X (np.ndarray): feature matrix [num_patients × num_features]
        y (np.ndarray): label matrix [num_patients × num_classes]
        pids (list): patient IDs
    """

    pred_cols = [f'preds_{i}' for i in range(num_cls)]
    lbl_cols = [f'lbls_{i}' for i in range(num_cls)]

    feats, labels, pids = [], [], []

    for pid, group in df.groupby('pid'):
        preds = group[pred_cols].values
        lbls = group[lbl_cols].values.mean(axis=0)  # should be identical per patient

        # --- Core statistical features ---
        feat = np.concatenate([
            preds.mean(axis=0),     # mean of class probs
            preds.std(axis=0),      # std of class probs
            preds.min(axis=0),      # min prob per class
            preds.max(axis=0),      # max prob per class
            preds.mean(axis=0)**2,  # nonlinear term
        ])

        # --- Optional stage feature ---
        if use_stage:
            if 'stage' not in group.columns:
                raise ValueError("use_stage=True but 'stage' column not found in df.")

            # Take stage as numeric (assume it's constant per patient)
            stage_val = group['stage'].iloc[0]
            if isinstance(stage_val, str):
                # Convert string categories to numeric (optional)
                stage_val = pd.Categorical(df['stage']).codes[group['stage'].index[0]]
            feat = np.concatenate([feat, np.array([stage_val], dtype=float)])

        feats.append(feat)
        labels.append(lbls)
        pids.append(pid)

    X = np.vstack(feats)
    y = np.vstack(labels)
    return X, y, pids


def learnable_aggregation(train_df, num_cls=2, model_type='logreg', use_stage=False):
    """
    Learn optimal aggregation from training data and apply on validation data.
    """

    # --- 1. Extract patient-level features ---
    X_train, y_train, _ = make_patient_features(train_df, num_cls, use_stage=use_stage)
    y_train_label = y_train.argmax(axis=1)

    # --- 2. Scale features ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # --- 3. Train meta-aggregator ---
    if model_type == 'logreg':
        meta_model = LogisticRegression(max_iter=1000)
    elif model_type == 'rf':
        meta_model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_type == 'knn':
        meta_model = KNeighborsClassifier(5, weights='distance')
    elif model_type == 'svc':
        meta_model = SVC(kernel='rbf', gamma='auto', probability=True)
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose from 'logreg', 'rf', 'knn', 'svc'.")

    meta_model.fit(X_train, y_train_label)

    return scaler, meta_model

def pre_aggregate(scaler, meta_model, val_df, use_stage=False):

    X_val, y_val, val_pids = make_patient_features(val_df, num_cls, use_stage=use_stage)
    y_val_label = y_val.argmax(axis=1)
    X_val = scaler.transform(X_val)

    # --- 4. Predict on validation ---
    val_preds = meta_model.predict_proba(X_val)

    # --- 5. Evaluate ---
    auc = roc_auc_score(y_val_label, val_preds[:, 1]) if num_cls == 2 else None
    acc = accuracy_score(y_val_label, val_preds.argmax(axis=1))
    bacc = balanced_accuracy_score(y_val_label, val_preds.argmax(axis=1))

    y_event_time = [df_l.loc[df_l['ID'] == int(x//10), 'Follow-up (days)'].to_list()[0] for x in val_pids]
    y_event = [df_l.loc[df_l['ID'] == int(x//10), 'Dead/Alive'].to_list()[0] == 'Dead' for x in val_pids]
    all_cindex = concordance_index_censored(y_event, y_event_time, [i[0] for i in val_preds])[0]

    print(f"Accuracy: {acc:.4f}", end='\t')
    print(f"Balanced Accuracy: {bacc:.4f}", end='\t')
    if auc:
        print(f"AUC: {auc:.4f}")
    else:
        print()

    # Return patient-level validation dataframe
    val_out = pd.DataFrame({
        'pid': val_pids,
        'pred_0': val_preds[:, 0],
        'pred_1': val_preds[:, 1],
        'true': y_val_label,
        'acc': acc,
        'bacc': bacc,
        'auc': auc,
        'cindex': all_cindex,

    })
    return val_out

stages0 = p.STAGE_CLASS_0

for method_name in p.LEARNING_METHODS: #['svc', 'rf', 'logreg', 'knn']:

    for use_stage in  p.ADD_STAGE: #[True, False]:
        print(f'\nMethod {method_name} {"with" if use_stage else "without"} stage:')

        fig, ax = plt.subplots()

        train_out, val_out, test_out = [], [], []

        for split_num in range(num_runs):
            path_train = os.path.join(path_complete, p.HBEST_FILE.format('train', split_num))
            path_val = os.path.join(path_complete, p.HBEST_FILE.format('val', split_num))
            path_test= os.path.join(path_complete, p.HBEST_FILE.format('test', split_num))
            train_df = pd.read_csv(path_train)
            train_df['stage'] = train_df['pid'].map(lambda x: 0 if df_l.loc[df_l['ID'] == int(x//10), p.STAGE_COLUMN].values[0] in stages0 else 1)
            val_df = pd.read_csv(path_val)
            val_df['stage'] = val_df['pid'].map(lambda x: 0 if df_l.loc[df_l['ID'] == int(x//10), p.STAGE_COLUMN].values[0] in stages0 else 1)
            test_df = pd.read_csv(path_test)
            test_df['stage'] = test_df['pid'].map(lambda x: 0 if df_l.loc[df_l['ID'] == int(x//10), p.STAGE_COLUMN].values[0] in stages0 else 1)

            for df in [train_df, val_df, test_df]:
                pred = df[[f'preds_{i}' for i in range(num_cls)]].values
                pred = [softmax(j) for j in pred]
                df[[f'preds_{i}' for i in range(num_cls)]] = pred

            scaler, meta_model = learnable_aggregation(
                train_df=pd.concat([train_df, val_df]),
                num_cls=num_cls,
                use_stage=use_stage,
                model_type=method_name,

            )
            print('Train: ', end='\t')
            train_out.append(pre_aggregate(scaler, meta_model, train_df, use_stage=use_stage,))

            print('Val:   ', end='\t')
            val_out.append(pre_aggregate(scaler, meta_model, val_df, use_stage=use_stage,))

            print('Test:  ', end='\t')
            test_out.append(pre_aggregate(scaler, meta_model, test_df, use_stage=use_stage,))

        plt.title(f'{method_name} {"- Stage" if use_stage else ""}')
        display = RocCurveDisplay.from_predictions(
            pd.concat(test_out)['true'],
            pd.concat(test_out)['pred_1'],
            ax=ax,
            plot_chance_level=True,#not split_num,
            curve_kwargs={'c':'blue'}
        )


        tmp_dict = {'C-Index':[float(i['cindex'][0]) for i in test_out],
                    'ROC':[float(i['auc'][0]) for i in test_out]}
        results_summary[f'{method_name}{"_stage" if use_stage else ""}'] = tmp_dict

        fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=False)

        axes[0].boxplot([[i.loc[0, 'auc'] for i in train_out],
                     [i.loc[0, 'auc'] for i in val_out],
                     [i.loc[0, 'auc'] for i in test_out]],
                    tick_labels=['train', 'val', 'test'])
        axes[0].plot([0.5, 3.5], [0.5, 0.5], 'grey', linestyle='--')
        axes[0].set_ylabel('auc')
        axes[0].grid()
        axes[0].set_ylim((0,1))



        axes[1].boxplot([[i.loc[0, 'cindex'] for i in train_out],
                     [i.loc[0, 'cindex'] for i in val_out],
                     [i.loc[0, 'cindex'] for i in test_out]],
                    tick_labels=['train', 'val', 'test'])
        axes[1].plot([0.5, 3.5], [0.5, 0.5], 'grey', linestyle='--')
        axes[1].set_ylabel('cindex')
        axes[1].grid()
        axes[1].set_ylim((0,1))
        plt.title(f'{method_name}{"_stage" if use_stage else ""}')
        plt.tight_layout()

        showplot()


        test_km = pd.concat(test_out)
        test_km['pred_all'] = test_km['pred_1'].map(lambda x: 'low' if x<0.5 else 'high')
        test_km['time'] = test_km['pid'].map(lambda x: df_l.at[int(x//10), 'time'])
        test_km['event'] = test_km['pid'].map(lambda x: df_l.at[int(x//10), 'event'])

        # Kaplan Meier

        # Initialize KM fitter
        kmf = KaplanMeierFitter()

        # Plot
        plt.figure(figsize=(8,6))

        for group_name, grouped_df in test_km.groupby('pred_all'):
            kmf.fit(
                durations=grouped_df["time"],
                event_observed=grouped_df["event"],
                label=f"predicted survival = {group_name}"
            )
            kmf.plot_survival_function(ci_show=True)#, show_censors=True)
            plt.plot([1730, 1730], [0, 1], '--k')
            plt.plot([0, 5000], [0.5, 0.5], '--k')

        plt.title("Kaplan–Meier Survival Curve by predicted survival")
        plt.xlabel("Follow-up time (days)")
        plt.ylabel("Survival probability")
        plt.legend(title='Predicted Survival')
        plt.grid(True)
        showplot()
        print()



df_out = pd.DataFrame.from_dict({i:results_summary[i]['ROC'] for i in results_summary})
print("==== AUROC ====")
print(df_out)
print("Summary")
df_stats = pd.DataFrame({
    "mean": df_out.mean(numeric_only=True),
    "std": df_out.std(numeric_only=True)
}).T
print(df_stats)
print()
df_out.to_csv(os.path.join(path_complete, 'all_auroc_0.csv'),float_format='%.8f')
df_stats.to_csv(os.path.join(path_complete, 'stats_auroc_0.csv'),float_format='%.8f')

df_out = pd.DataFrame.from_dict({i:results_summary[i]['C-Index'] for i in results_summary})
print("=== C-Index ===")
print(df_out)
print("Summary")
df_stats = pd.DataFrame({
    "mean": df_out.mean(numeric_only=True),
    "std": df_out.std(numeric_only=True)
}).T
print(df_stats)
print()
df_out.to_csv(os.path.join(path_complete, 'all_cindex_0.csv'),float_format='%.8f')
df_stats.to_csv(os.path.join(path_complete, 'stats_cindex_0.csv'),float_format='%.8f')

