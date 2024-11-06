import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from ot import emd

def match(
    data, group_var, 
    g1_label, g2_label, 
    targets,
    targets2encode=None,
    distance = "euclidean"):

    # encode targets
    data = data.copy() 
    if targets2encode:
        for t in targets2encode:
            data.loc[:, t] = data[t].astype("category").cat.codes

    g1 = data.loc[data[group_var]==g1_label]
    g2 = data.loc[data[group_var]==g2_label]

    print(g1.shape, g2.shape)

    # Compute the cost matrix (Euclidean distance)
    cost_matrix = cdist(g1[targets], g2[targets], metric='cityblock')

    n1, n2 = len(g1), len(g2)
    a = np.ones(n1) / n1  # Uniform distribution for group_1
    b = np.ones(n2) / n2  # Uniform distribution for group_2

    # Solve the optimal transport problem
    transport_plan = emd(a, b, cost_matrix)

    return transport_plan, cost_matrix, g1, g2

def inspect_matching(
    g1, g2, transport_plan, cost_matrix, min_samples_left=None, 
    plot=True, show_plot=False, save_path=None, add_info=""):

    n_left_g1 = []
    n_left_g2 = []
    total_costs = []
    thresholds = sorted(np.unique(transport_plan))
    for i, th in enumerate(thresholds):
        significant_samples_left = np.argwhere(transport_plan > th)
        samples_left_group_1 = np.unique(significant_samples_left[:, 0])
        samples_left_group_2 = np.unique(significant_samples_left[:, 1])
        selected_samples_1 = g1.iloc[samples_left_group_1]
        selected_samples_2 = g2.iloc[samples_left_group_2]
        # return selected_samples
        n_left_g1.append(selected_samples_1.shape[0])
        n_left_g2.append(selected_samples_2.shape[0])
        total_costs.append(
            pd.DataFrame(transport_plan).iloc[
                np.unique(significant_samples_left[:,0]), 
                np.unique(significant_samples_left[:,1])].sum().sum()
                )
    res_th = pd.DataFrame(
        {
            "th":thresholds,
            "n_left_1":n_left_g1,
            "n_left_2":n_left_g2
            }
            ).sort_values("n_left_1")
    # select th based on min samples
    th_sel = res_th[res_th["n_left_1"] >= min_samples_left].head(1)["th"]

    ## Plotting  
    if plot:
        fig, axs = plt.subplots(1,2, sharey=True)
        axs[0].scatter(
            x=n_left_g1,
            y=thresholds, 
            lw=0.5, ec="k", alpha=.5, c="tab:blue")
        axs[0].plot(n_left_g1, thresholds)
        axs[1].scatter(
            x=total_costs,
            y=thresholds,
            lw=0.5, ec="k", alpha=.5, c="tab:blue")
        axs[1].plot(total_costs, thresholds)
        if min_samples_left != None:
            axs[0].axvline(
                min_samples_left, 
                c="g", ls="--", lw=3,
                label=f"Min samples (n={min_samples_left})")
        fig.supylabel("Index Threshold", fontweight="bold")
        axs[0].set_xlabel("N samples left in G1", fontweight="bold")
        axs[1].set_xlabel("Total costs(-)", fontweight="bold")
        plt.tight_layout()
        fig.legend()
        if save_path != None:
            plt.savefig(os.path.join(save_path, f'matching_plot_{add_info}.png'),dpi=400)
        if show_plot:
            plt.show()
        else:
            plt.close()
    return res_th