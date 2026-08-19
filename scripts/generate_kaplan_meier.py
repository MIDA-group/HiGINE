#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a side-by-side Kaplan-Meier survival plot comparing two
train_coregraphs.py result sets (e.g. with vs. without cancer stage as a
node feature), aggregated to patient level.
"""

import os
import argparse
import yaml
from types import SimpleNamespace
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import auc, roc_curve
from sksurv.metrics import concordance_index_censored


# load params
params_file = './settings/mif/generate_kaplan_meier.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--parameters-file', default=params_file, type=str, help='file where the parameters for the figure are stored')
args = parser.parse_args()
params_file = args.parameters_file

with open(params_file, 'r') as params:
    p = SimpleNamespace(**yaml.safe_load(params))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


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
    if num_cls is None:
        pred_cols = [c for c in df.columns if c.startswith('preds_')]
        num_cls = len(pred_cols)

    pred_cols = [f'preds_{i}' for i in range(num_cls)]
    lbl_cols = [f'lbls_{i}' for i in range(num_cls) if f'lbls_{i}' in df.columns]

    grouped = df.groupby('pid')
    pid_list, pred_list, true_list = [], [], []

    for pid, group in grouped:
        preds = group[pred_cols].values

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


def get_agg_preds_df(core_path, folds, df_l):
    pred_dfs = []

    for i in range(folds):
        pred_dfs.append(pd.read_csv(os.path.join(core_path, f'predictions_hbest_test_split_{i}.csv')))

    preds_df = pd.concat(pred_dfs)

    num_cls = 2
    pred = preds_df[[f'preds_{i}' for i in range(num_cls)]].values
    pred = [softmax(j) for j in pred]
    preds_df[[f'preds_{i}' for i in range(num_cls)]] = pred

    agg_preds_df = aggregate_pid(preds_df, "mean", num_cls)

    pred = agg_preds_df[[f'preds_{i}' for i in range(num_cls)]].values
    true = agg_preds_df[[f'lbls_{i}' for i in range(num_cls)]].values

    fpr, tpr, _ = roc_curve([i[1] for i in true], [i[1] for i in pred])
    print("AUROC: ", auc(fpr, tpr))

    agg_preds_df["time"] = agg_preds_df.pid.map(lambda x: df_l.loc[df_l['ID'] == int(x//10), 'Follow-up (days)'].to_list()[0])
    agg_preds_df["event"] = agg_preds_df.pid.map(lambda x: df_l.loc[df_l['ID'] == int(x//10), 'Dead/Alive'].to_list()[0] == 'Dead')
    print("C-index: ", concordance_index_censored(agg_preds_df["event"], agg_preds_df["time"], [i[0] for i in pred])[0])

    agg_preds_df["pred_class"] = (agg_preds_df["preds_1"] > agg_preds_df["preds_0"]).astype(int)

    return agg_preds_df


def kmf_fit_plot(agg_preds, ax, kmf, labels):
    for cls, subset in agg_preds.groupby("pred_class"):
        kmf.fit(subset["time"], subset["event"], label=labels[cls])
        kmf.plot_survival_function(
            ax=ax,
            linewidth=10,
            ci_show=True,
            show_censors=True,
            censor_styles={'marker': '+', 'ms': 30, 'mew': 5},
        )

    cox_df = agg_preds.copy()
    cox_df["pred_class"] = 1 - cox_df["pred_class"].astype(int)  # Inverting label, to get increased hazard if in short-term group
    cox = CoxPHFitter()
    cox.fit(cox_df[["time", "event", "pred_class"]], duration_col="time", event_col="event")
    hr = np.exp(cox.params_["pred_class"])
    print(cox.hazard_ratios_)

    g1 = agg_preds[agg_preds["pred_class"] == 0]
    g2 = agg_preds[agg_preds["pred_class"] == 1]
    result = logrank_test(g1["time"], g2["time"], g1["event"], g2["event"])
    pval = result.p_value
    print('pval: ', pval)
    return result, pval, hr


def main():
    df_l = pd.read_csv(p.LABEL_DATA)
    df_l.index = df_l['ID']
    df_l["event"] = df_l["Dead/Alive"].map(lambda x: x == "Dead")
    df_l["time"] = df_l["Follow-up (days)"]

    core_results = p.CORE_RESULTS
    agg_preds_dfs = [get_agg_preds_df(cr['path'], p.FOLDS, df_l) for cr in core_results]

    pred_labels = {
        0: "Predicted short-term survival",
        1: "Predicted long-term survival",
    }

    kmf = KaplanMeierFitter()

    fig, axs = plt.subplots(1, len(agg_preds_dfs), figsize=(20 * len(agg_preds_dfs), 20))
    if len(agg_preds_dfs) == 1:
        axs = [axs]

    plt.rcParams.update({
        "text.usetex": False,            # no external LaTeX needed
        "font.family": 'DejaVu Sans',
        "font.serif": ["DejaVu Serif"],  # bundled serif font, always available
        "mathtext.fontset": "cm",        # use Computer Modern for math
        "mathtext.rm": "serif",
    })

    handles, labels = None, None
    for ax, agg_preds_df, core_result in zip(axs, agg_preds_dfs, core_results):
        result, pval, hr = kmf_fit_plot(agg_preds_df, ax, kmf, pred_labels)

        textstr = f"HR = {hr:.2f}\np = {pval:.1e}"
        ax_pos = ax.get_position()
        fig.text(
            ax_pos.x0 + 0.02, 0.02, textstr,
            fontsize=60,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=1)
        )

        fig.text(
            (ax_pos.x0 + ax_pos.x1) / 2, 1.02, core_result['label'],
            fontsize=80,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.6)
        )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()

    axs[0].set_ylabel("Overall Survival (%)", fontsize=70)
    for ax in axs:
        ax.set_xlabel("Time (days)", fontsize=70)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: int(y * 100)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_linewidth(10.0)
        ax.spines['left'].set_linewidth(10.0)

        ax.tick_params(axis='both', width=10.0, length=25, labelsize=50)

    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=70)

    plt.grid(False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(p.output_path), exist_ok=True)
    plt.savefig(p.output_path, dpi=150, bbox_inches='tight')
    print(f'Saved figure to {p.output_path}')


if __name__ == '__main__':
    main()
