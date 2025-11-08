# -*- coding: utf-8 -*-
"""
UCS (Unconfined Compressive Strength) Dataset Loader

This module provides functions to load various UCS datasets from different research papers
for geotechnical and materials science applications.

Features:
- Loads multiple UCS datasets from different sources
- Handles data preprocessing and feature engineering
- Provides train-test splitting capabilities
- Generates correlation heatmaps and statistical summaries
- Supports both regression and classification tasks
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration for plotting
pl.rc("text", usetex=True)
pl.rc("font", **{"family": "serif", "serif": ["Palatino"]})

# Compatibility fix for newer numpy versions
np.bool = np.bool_


def read_ucs(dataset, test_size=None, seed=None):
    """
    Main function to read UCS datasets based on dataset identifier.
    
    Parameters:
    -----------
    dataset : str
        Dataset identifier (D1-D10, S1, RCA, D0S)
    test_size : float, optional
        Proportion of dataset to include in test split
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing dataset information and splits
    """
    dataset_map = {
        "D1": lambda: read_gajurel(target="UCS", treatment="Lime", dataset=dataset, 
                                  test_size=test_size, seed=seed),
        "D2": lambda: read_gajurel(target="UCS", treatment="Cement", dataset=dataset, 
                                  test_size=test_size, seed=seed),
        "D3": lambda: read_ngo(dataset=dataset, test_size=test_size, seed=seed),
        "D4": lambda: read_priyadarshee(dataset=dataset, test_size=test_size, seed=seed),
        "D5": lambda: read_mozumder(dataset=dataset, test_size=test_size, seed=seed),
        "D6": lambda: read_taffese(dataset=dataset, test_size=test_size, seed=seed),
        "D7": lambda: read_tabarsa(dataset=dataset, test_size=test_size, seed=seed),
        "D8": lambda: read_mahmoodzadeh(dataset=dataset, test_size=test_size, seed=seed),
        "D9": lambda: read_wang(dataset=dataset, test_size=test_size, seed=seed),
        "D10": lambda: read_zhang(dataset=dataset, test_size=test_size, seed=seed),
        "RCA": lambda: read_yuan(dataset=dataset, test_size=test_size, seed=seed),
        "D0S": lambda: read_burroughs(dataset=dataset, test_size=test_size, seed=seed),
        "S1": lambda: read_jalal(dataset=dataset, test_size=test_size, seed=seed),
    }
    
    if dataset in dataset_map:
        return dataset_map[dataset]()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def read_burroughs(target="UCS", dataset=None, test_size=None, seed=None):
    """
    Load Burroughs dataset for bio-cemented sand samples.
    
    Reference: https://doi.org/10.1016/j.jclepro.2021.128205
    
    Features:
    - d50: Median grain size
    - Cu: Coefficient of uniformity  
    - e0: Initial void ratio
    - OD600: Optical density of bacterial suspension
    - Mu: Urea concentration
    - MCa: Calcium concentration
    - FCa: Calcium carbonate content
    - UCS: Unconfined compressive strength
    """
    fn = "./data/data_burroughs/whole.txt"
    X = pd.read_csv(fn, header=None)
    X = X.values.ravel().reshape(-1, 26)
    cols = "d50 Cu e0 OD600 Mu MCa FCa UCS".split(" ")
    X = pd.DataFrame(X, columns=cols)
    
    # Convert to float
    for c in X.columns:
        X[c] = X[c].astype(float)
    
    target_names = ["UCS"]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    # Handle categorical columns
    categorical_columns = []
    for cc in categorical_columns:
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
    
    # Create splits
    X_train, y_train = X[variable_names], X[target_names]
    X_test, y_test = pd.DataFrame([]), pd.DataFrame([])
    
    # Generate statistics
    n_samples, n_features = X_train.shape
    df_train = X_train.copy()
    df_train[target_names] = y_train
    stat_train = df_train.describe().T
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption=f"Basic statistics for dataset {dataset}.",
    )
    
    return _create_regression_data_dict(
        dataset=dataset,
        variable_names=variable_names,
        target_names=target_names,
        X_train=X_train.values,
        y_train=y_train.values.T,
        X_test=X_test.values,
        y_test=[[]] * len(target_names),
        reference="https://doi.org/10.1016/j.jclepro.2021.128205"
    )


def read_wang(target="UCS", dataset=None, test_size=None, seed=None):
    """
    Load Wang et al. (2021) dataset for bio-cemented sand.
    
    Reference: https://doi.org/10.1016/j.jclepro.2021.128205
    """
    fn = "./data/data_wang/wang2021.txt"
    X = np.loadtxt(fn).reshape(-1, 8)
    cols = "d50 Cu e0 OD600 Mu MCa FCa UCS".split(" ")
    X = pd.DataFrame(X, columns=cols)
    
    for c in X.columns:
        X[c] = X[c].astype(float)
    
    target_names = ["UCS"]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    # Create train-test splits
    X_train, X_test, y_train, y_test = _create_splits(
        X[variable_names].values, 
        X[target_names].values, 
        test_size, 
        seed
    )
    
    # Generate visualization and statistics
    _generate_correlation_heatmap(X, variable_names, target_names, dataset)
    n_samples, n_features = X_train.shape
    
    return _create_regression_data_dict(
        dataset=dataset,
        variable_names=variable_names,
        target_names=target_names,
        X_train=X_train,
        y_train=y_train.reshape(1, -1),
        X_test=X_test,
        y_test=y_test.reshape(1, -1),
        reference="https://doi.org/10.1016/j.jclepro.2021.128205"
    )


def read_zhang(target="UCS", dataset=None, test_size=None, seed=None):
    """
    Load Zhang et al. (2022) dataset.
    
    Reference: https://doi.org/10.1016/j.jmrt.2022.02.076
    """
    fn = "./data/data_zhang/zhang2022.csv"
    X = pd.read_csv(fn, header=0)
    X.drop(["Specimen"], axis=1, inplace=True)
    
    for c in X.columns:
        X[c] = X[c].astype(float)
    
    target_names = ["UCS"]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    # Create train-test splits
    X_train, X_test, y_train, y_test = _create_splits(
        X[variable_names].values,
        X[target_names].values,
        test_size,
        seed
    )
    
    # Generate visualization and statistics
    _generate_correlation_heatmap(X, variable_names, target_names, dataset)
    
    return _create_regression_data_dict(
        dataset=dataset,
        variable_names=variable_names,
        target_names=target_names,
        X_train=X_train,
        y_train=y_train.reshape(1, -1),
        X_test=X_test,
        y_test=y_test.reshape(1, -1),
        reference="https://doi.org/10.1016/j.jmrt.2022.02.076"
    )


def read_priyadarshee(target="UCS", dataset=None, test_size=None, seed=None):
    """
    Load Priyadarshee et al. dataset for stabilized soils.
    
    Reference: https://doi.org/10.3390/ma15082823
    """
    fn = "./data/data_priyadarshee/data_priyadarshee.csv"
    X = pd.read_csv(fn, header=0)
    X.drop(["Number", "Predicted"], axis=1, inplace=True)
    
    for c in X.columns.drop("UCS"):
        X[c] = X[c].astype(float)
    
    # Rename columns for clarity
    X.columns = ["C", "PondAsh", "RiceHusk", "Cement", "Curing", "UCS"]
    
    target_names = ["UCS"]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    # Create train-test splits
    X_train, X_test, y_train, y_test = _create_splits(
        X[variable_names].values,
        X[target_names].values,
        test_size,
        seed
    )
    
    # Generate visualization and statistics
    _generate_correlation_heatmap(X, variable_names, target_names, dataset)
    
    return _create_regression_data_dict(
        dataset="D4",
        variable_names=variable_names,
        target_names=target_names,
        X_train=X_train,
        y_train=y_train.reshape(1, -1),
        X_test=X_test,
        y_test=y_test.reshape(1, -1),
        reference="https://doi.org/10.3390/ma15082823"
    )


def read_yuan(target="CS", dataset=None, test_size=None, seed=None, plot=False):
    """
    Load Yuan et al. (2022) dataset for recycled aggregate concrete.
    
    Reference: https://www.mdpi.com/1996-1944/15/8/2823
    
    Features include various concrete mixture parameters.
    """
    fn = "./data/data_yuan/yuan2022_recycled_aggregate_concrete.txt"
    X = np.loadtxt(fn).reshape(-1, 14)
    cols = "weffc acr rcar pcs nmrcas nmnas bdrca bdna warca wana larca lana CS FS".split(" ")
    X = pd.DataFrame(X, columns=cols)
    
    target_names = [target]
    
    if target == "FS":
        X = X[X["FS"] != 0]
    
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    # Create train-test splits
    X_train, X_test, y_train, y_test = _create_splits(
        X[variable_names].values,
        X[target_names].values,
        test_size,
        seed
    )
    
    return _create_regression_data_dict(
        dataset=dataset,
        variable_names=variable_names,
        target_names=target_names,
        X_train=X_train,
        y_train=y_train.reshape(1, -1),
        X_test=X_test,
        y_test=y_test.reshape(1, -1),
        reference="https://www.mdpi.com/1996-1944/15/8/2823"
    )


def read_ngo(target="UCS", dataset=None, test_size=None, seed=None):
    """
    Load Ngo et al. dataset for soil stabilization.
    
    Reference: https://doi.org/10.3390/app11041949
    """
    fn = "./data/data_ngo/ngo.txt"
    X = np.loadtxt(fn).reshape(-1, 16)
    cols = "No D We Cc Cp S Mc T Ac Di L A V M De qu".split(" ")
    X = pd.DataFrame(X, columns=cols)
    
    # Feature engineering
    X["UCS"] = X["qu"] / 1e3  # Convert to MPa
    X.drop(["qu", "No", "L"], axis=1, inplace=True)
    
    target_names = ["UCS"]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    # Create train-test splits
    X_train, X_test, y_train, y_test = _create_splits(
        X[variable_names].values,
        X[target_names].values,
        test_size,
        seed
    )
    
    # Generate visualization and statistics
    _generate_correlation_heatmap(X, variable_names, target_names, dataset, figsize=(7, 6))
    
    return _create_regression_data_dict(
        dataset=dataset,
        variable_names=variable_names,
        target_names=target_names,
        X_train=X_train,
        y_train=y_train.reshape(1, -1),
        X_test=X_test,
        y_test=y_test.reshape(1, -1),
        reference="https://doi.org/10.3390/app11041949"
    )


# Additional dataset reader functions follow similar pattern...
# read_jalal, read_taffese, read_mahmoodzadeh, read_mozumder, read_tabarsa, read_gajurel


def _create_splits(X, y, test_size, seed):
    """Helper function to create train-test splits."""
    if test_size == 0 or test_size is None:
        return X, y, pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=seed
        )
        return X_train, X_test, y_train.ravel(), y_test.ravel()


def _generate_correlation_heatmap(X, variable_names, target_names, dataset, figsize=(5, 4)):
    """Helper function to generate correlation heatmaps."""
    df = X[variable_names + target_names].copy()
    
    pl.figure(figsize=figsize)
    corr = df.corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap="PuOr",
    )
    heatmap.set_title(
        f"{dataset}: Correlation Heatmap", fontdict={"fontsize": 12}, pad=12
    )
    pl.savefig(f"{dataset}_heatmap_correlation.png", bbox_inches="tight", dpi=300)
    pl.show()


def _create_regression_data_dict(dataset, variable_names, target_names, X_train, y_train, 
                               X_test, y_test, reference, task="regression"):
    """Helper function to create standardized regression data dictionary."""
    n_samples, n_features = X_train.shape
    
    return {
        "task": task,
        "name": dataset,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "targets": target_names,
        "true_labels": None,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": reference,
        "items": None,
        "normalize": None,
    }


def lhsu(xmin, xmax, nsample):
    """
    Latin Hypercube Sampling with uniform distribution.
    
    Parameters:
    -----------
    xmin : array-like
        Lower bounds for each variable
    xmax : array-like  
        Upper bounds for each variable
    nsample : int
        Number of samples to generate
    
    Returns:
    --------
    ndarray
        Latin hypercube samples
    """
    nvar = len(xmin)
    ran = np.random.rand(nsample, nvar)
    s = np.zeros((nsample, nvar))
    
    for j in range(nvar):
        idx = np.random.permutation(nsample)
        P = (idx.T - ran[:, j]) / nsample
        s[:, j] = xmin[j] + P * (xmax[j] - xmin[j])
    
    return s


if __name__ == "__main__":
    # Example usage
    ds_names = ["D" + str(i + 1) for i in range(10)]
    ds_names += ["S1", "RCA"]
    
    for d in ds_names:
        try:
            D = read_ucs(d)
            print(f"{D['name']}: {D['n_samples']} samples")
            print(f"Features: {D['feature_names']}")
            print("-" * 50)
        except Exception as e:
            print(f"Error loading {d}: {e}")