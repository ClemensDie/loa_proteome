import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os 

from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

def plot_volcano_limma(
    data_volcano, save_path=None, show_plot=True, 
    add_info="", xlabel = "coef"):

    signif_features = data_volcano[(data_volcano["adj.P.Val"]< 0.05) & ((data_volcano["logFC"] < -0.1) | (data_volcano["logFC"] > 0.1))].index
    print(len(signif_features))
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.scatter(
        x=data_volcano.loc[(data_volcano[f"adj.P.Val"] >= 0.05), f"logFC"], 
        y=-np.log10(data_volcano.loc[(data_volcano[f"adj.P.Val"] >= 0.05), f"adj.P.Val"]), c="k", alpha=0.5)
    ax.scatter(
        x=data_volcano.loc[signif_features, f"logFC"], 
        y=-np.log10(data_volcano.loc[signif_features, f"adj.P.Val"]), c="#FF3131", alpha=0.99, edgecolor="gray")

    ax.set_xlabel("$\\bf{Log_{2} FC}$", fontdict={"size":14})
    ax.set_ylabel("$\\bf{-log_{10} (p_{adj.})}$", fontdict={"size":14, "weight":"bold"})        

    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", fontsize=9)
    ax.axhline(1.3, c="gray", ls="--")
    ax.axvline(0.1, c="gray", ls="--")
    ax.axvline(-0.1, c="gray", ls="--")
    plt.grid()
    plt.tight_layout()

    if save_path != None:
        plt.savefig(os.path.join(save_path, f'{add_info}.png'),dpi=400)
    if show_plot:
        plt.show()
    else:
        plt.close()
   
def plot_fc_diff(x,y, xlabel, ylabel, save_path):
    x = x.dropna()
    y = y.dropna()

    common_idx = list(set(x.index) & set(y.index))
    x = x.loc[common_idx]
    y = y.loc[common_idx]

    diff = np.abs(x.copy().dropna().values,y.copy().values)
    diff = (pd.Series(MinMaxScaler().fit_transform(diff.reshape(-1, 1)).ravel()))
    s_rho, p = spearmanr(x,y)

    # return diff
    fig, ax = plt.subplots(1,1, figsize=(4,4))
    for i in range(len(x)):
        ax.scatter(x[i],y[i], c="k", alpha=diff[i], s=(diff[i]+0.1)*100)
    ax.axhline(0, c="k", ls="--")
    ax.axvline(0, c="k", ls="--")
    ax.set_xlabel("LogFC " + xlabel, fontweight="bold")
    ax.set_ylabel("LogFC " + ylabel, fontweight="bold")
    plt.legend(["s$_{\\rho}$: " + f"{s_rho:.2f}, p: {p:.2f}"], loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{xlabel}_{ylabel}.png"), dpi=400)