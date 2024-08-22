import os
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns


def match(
    data, group_var, 
    g1_label, g2_label, 
    targets,
    targets2encode):

    """
    Perform optimal transport matching between two groups in a dataset based on specified features.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing the samples to be matched.
    group_var : str
        The column name in `data` that indicates group membership.
    g1_label : any
        The label indicating the first group in `group_var`.
    g2_label : any
        The label indicating the second group in `group_var`.
    targets : list of str
        List of column names in `data` to be used as features for matching.
    targets2encode : list of str
        List of column names in `targets` that contain categorical variables.
        These will be encoded as integer codes before matching.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        The result of the linear programming problem, containing information about
        the optimization process, including the optimal solution.
    transport_plan : numpy.ndarray
        A matrix of shape (n1, n2) where n1 and n2 are the number of samples in 
        group 1 and group 2, respectively. The matrix indicates the optimal transport 
        plan, i.e., how much of each sample in group 1 is matched to each sample in group 2.

    Mathematical Background
    -----------------------
    The function solves an optimal transport problem, which seeks to match two groups of samples 
    (group 1 and group 2) in a way that minimizes the total transportation cost.

    Let:
    - `g1` be the subset of `data` corresponding to `g1_label` (group 1).
    - `g2` be the subset of `data` corresponding to `g2_label` (group 2).
    - `C` be the cost matrix of shape (n1, n2) where `C[i, j]` is the cost (or distance) of transporting 
      a unit of mass from sample `i` in group 1 to sample `j` in group 2. This cost is computed using 
      the Euclidean distance between the feature vectors of the samples.

    The objective is to find a transport plan `X` that minimizes the total cost:
    \[
    \min \sum_{i=1}^{n1} \sum_{j=1}^{n2} C_{ij} \cdot X_{ij}
    \]
    subject to:
    - \(\sum_{j=1}^{n2} X_{ij} = a_i\) for each `i` in group 1 (supply constraints),
    - \(\sum_{i=1}^{n1} X_{ij} = b_j\) for each `j` in group 2 (demand constraints),
    - \(X_{ij} \geq 0\) for all `i, j`.

    Here:
    - `a` is the supply vector for group 1 (uniform in this case, i.e., each sample contributes equally).
    - `b` is the demand vector for group 2 (also uniform).

    The linear programming problem is solved using the `linprog` function from `scipy.optimize`, 
    which finds the optimal transport plan `X` that satisfies the above constraints.

    The transport plan `X` indicates how much of each sample in group 1 is matched to each sample 
    in group 2. The resulting transport plan can be used to understand how the distributions of 
    the two groups align or differ.
    
    Example
    -------
    >>> data = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'age': [30, 40, 50, 35, 45, 55],
            'sex': ['M', 'F', 'M', 'M', 'F', 'M']
        })
    >>> match(data, group_var='group', g1_label='A', g2_label='B', 
              targets=['age', 'sex'], targets2encode=['sex'])
    
    """

    g1 = data.loc[data[group_var]==g1_label, targets]
    g2 = data.loc[data[group_var]==g2_label, targets]

    for t in targets2encode:
        g1.loc[:, t] = g1[t].astype("category").cat.codes
        g2.loc[:, t] = g2[t].astype("category").cat.codes

    cost_matrix = cdist(g1.values, g2.values, metric="euclidean")

    n1, n2 = len(g1), len(g2)
    a = np.ones(n1) / n1  # Uniform distribution for group_1
    b = np.ones(n2) / n2  # Uniform distribution for group_2

    c = cost_matrix.flatten()

    # Define Constraints
    A_eq = [] # Equality contraint
    # Supply contraints
    # Total amount of mass transported from each element in g1 should
    # equal the supply a_i for that element
    for i in range(n1):
        row = np.zeros(n1 * n2)
        row[i*n2:(i+1)*n2] = 1
        A_eq.append(row)
    # Demand contraints
    # Total amount of mass received by each element in g2 should
    # equal the demand b_j for that element
    for j in range(n2):
        row = np.zeros(n1 * n2)
        row[j::n2] = 1
        A_eq.append(row)
    A_eq = np.array(A_eq)

    b_eq = np.concatenate([a, b]) # right handside vector
    x_bounds = [(0, None) for _ in range(n1 * n2)] # all transport values have to be
    # Solve
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')
    # Reshape Solution into transport Plan
    transport_plan = result.x.reshape(n1, n2)

    return result, transport_plan

def inspect_matching(
    g1, transport_plan, cost_matrix, min_matches=None, 
    plot=True, show_plot=False, save_path=None, add_info=""):

    n_matches = []
    total_costs = []
    thresholds = sorted(np.unique(transport_plan))
    for i, th in enumerate(thresholds[::-1]):
        significant_matches = np.argwhere(transport_plan > th)
        best_matches_group_1 = np.unique(significant_matches[:, 0])
        selected_samples = g1.iloc[best_matches_group_1]
        n_matches.append(selected_samples.shape[0])
        total_costs.append(
            np.sum(
                transport_plan[
                    significant_matches]*cost_matrix[significant_matches]
                )
        )
    ## Plotting  
    if plot:
        fig, axs = plt.subplots(1,2, sharey=True)
        axs[0].scatter(
            x=n_matches,
            y=thresholds, 
            lw=0.5, ec="k", alpha=.5, c="tab:blue")
        axs[0].plot(n_matches, thresholds)
        axs[1].scatter(
            x=total_costs,
            y=thresholds,
            lw=0.5, ec="k", alpha=.5, c="tab:blue")
        axs[1].plot(total_costs, thresholds)
        if min_matches != None:
            axs[0].axvline(
                min_matches, 
                c="g", ls="--", lw=3,
                label=f"Min samples (n={min_matches})")
        fig.supylabel("Index Threshold", fontweight="bold")
        axs[0].set_xlabel("N samples Left in G1", fontweight="bold")
        axs[1].set_xlabel("Total costs(-)", fontweight="bold")
        plt.tight_layout()
        fig.legend()
        if save_path != None:
            plt.savefig(os.path.join(save_path, f'matching_plot_{add_info}.png'),dpi=400)
        if show_plot:
            plt.show()
        else:
            plt.close()