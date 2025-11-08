# -*- coding: utf-8 -*-
"""
Organized and Documented Source Code for UCS Dataset Readers

This script provides functions to load and process various datasets related to 
Unconfined Compressive Strength (UCS) and other geotechnical properties from 
different research papers. The datasets are prepared for machine learning tasks, 
primarily regression.

Key Components:
- Import necessary libraries for data handling, visualization, and ML.
- Define a dispatcher function `read_ucs` to load specific datasets.
- Individual reader functions for each dataset, which:
  - Load data from files (CSV, TXT, XLSX).
  - Clean and preprocess data (e.g., drop columns, handle types, encode categoricals).
  - Optionally split into train/test sets.
  - Compute statistics and optionally plot correlation heatmaps.
  - Return a standardized dictionary with data, metadata, and references.
- A utility function `lhsu` for Latin Hypercube Sampling.
- Main block for example usage.

Dependencies:
- numpy
- pandas
- seaborn
- matplotlib (via pylab)
- scikit-learn (for train_test_split and LabelEncoder)

Usage Example:
    D = read_ucs('D1', test_size=0.2, seed=42)
    print(D['name'], D['n_samples'])
    print(D['feature_names'])

Notes:
- Datasets are stored in './data/' directory with subfolders.
- Some functions have optional plotting (disabled by default).
- Target is typically 'UCS' (regression), but can be classification in some cases.
- Code assumes no internet access for additional package installs.

Author: [Your Name or Original Author]
Date: November 08, 2025
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Fix for deprecated np.bool in newer NumPy versions
np.bool = np.bool_

# Set plotting defaults
pl.rc('text', usetex=True)
pl.rc('font', **{'family': 'serif', 'serif': ['Palatino']})


def read_ucs(dataset, test_size=None, seed=None):
    """
    Dispatcher function to load a specific UCS dataset.

    Parameters:
    - dataset (str): Dataset identifier (e.g., 'D1' for Lime-treated, 'D2' for Cement-treated).
    - test_size (float or None): Fraction of data for test set (0-1). If None or 0, no split.
    - seed (int or None): Random seed for train-test split.

    Returns:
    - dict: Standardized dataset dictionary for ML tasks.
    """
    if dataset == 'D1':
        return read_gajurel(target='UCS', treatment='Lime', dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D2':
        return read_gajurel(target='UCS', treatment='Cement', dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D3':
        return read_ngo(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D4':
        return read_priyadarshee(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D5':
        return read_mozumder(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D6':
        return read_taffese(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D7':
        return read_tabarsa(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D8':
        return read_mahmoodzadeh(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D9':
        return read_wang(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D10':
        return read_zhang(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'RCA':
        return read_yuan(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'D0S':
        return read_burroughs(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == 'S1':
        return read_jalal(dataset=dataset, test_size=test_size, seed=seed)
    raise ValueError(f"Unknown dataset: {dataset}")


def read_burroughs(target='UCS', dataset=None, test_size=None, seed=None):
    """
    Load and process Burroughs dataset for bio-cemented sand samples.

    Parameters:
    - target (str): Target column (default 'UCS').
    - dataset (str or None): Dataset name for metadata.
    - test_size (float or None): Test set fraction.
    - seed (int or None): Random seed.

    Returns:
    - dict: Dataset dictionary.
    """
    fn = './data/data_burroughs/whole.txt'
    X = pd.read_csv(fn, header=None)
    X = X.values.ravel().reshape(-1, 8)  # Adjusted to match actual columns
    cols = 'd50 Cu e0 OD600 Mu MCa FCa UCS'.split()
    X = pd.DataFrame(X, columns=cols)
    
    for c in X.columns:
        X[c] = X[c].astype(float)
    
    target_names = [target]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    # No categorical columns in this dataset
    X_train, y_train = X[variable_names], X[target_names]
    X_test, y_test = pd.DataFrame([], columns=variable_names), pd.DataFrame([], columns=target_names)
    
    # Correlation heatmap (commented out)
    # df = X.copy()
    # pl.figure(figsize=(5, 4))
    # corr = df.corr().round(2)
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    # pl.savefig(dataset + '_heatmap_correlation.png', bbox_inches='tight', dpi=300)
    # pl.show()
    
    n_samples, n_features = X_train.shape
    df_train = X_train.copy()
    df_train[target_names] = y_train
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': dataset,
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train.values,
        'y_train': y_train.values.T,
        'X_test': X_test.values,
        'y_test': [[]] * len(target_names),
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.1016/j.jclepro.2021.128205",
        'items': None,
        'normalize': None,
    }


def read_wang(target='UCS', dataset=None, test_size=None, seed=None):
    """
    Load and process Wang (2021) dataset for bio-cemented sand.

    Parameters: Same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_wang/wang2021.txt'
    X = np.loadtxt(fn).reshape(-1, 8)
    cols = 'd50 Cu e0 OD600 Mu MCa FCa UCS'.split()
    X = pd.DataFrame(X, columns=cols)
    
    for c in X.columns:
        X[c] = X[c].astype(float)
    
    target_names = [target]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target_names].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': dataset,
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.1016/j.jclepro.2021.128205",
        'items': None,
        'normalize': None,
    }


def read_zhang(target='UCS', dataset=None, test_size=None, seed=None):
    """
    Load and process Zhang (2022) dataset.

    Parameters: Same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_zhang/zhang2022.csv'
    X = pd.read_csv(fn, header=0)
    X.drop(['Specimen'], axis=1, inplace=True)
    for c in X.columns:
        X[c] = X[c].astype(float)
    
    target_names = [target]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target_names].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': dataset,
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.1016/j.jmrt.2022.02.076",
        'items': None,
        'normalize': None,
    }


def read_priyadarshee(target='UCS', dataset=None, test_size=None, seed=None):
    """
    Load and process Priyadarshee dataset.

    Parameters: Same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_priyadarshee/data_priyadarshee.csv'
    X = pd.read_csv(fn, header=0)
    X.drop(['Number', 'Predicted'], axis=1, inplace=True)
    for c in X.columns.drop('UCS'):
        X[c] = X[c].astype(float)
    X.columns = ['C', 'PondAsh', 'RiceHusk', 'Cement', 'Curing', 'UCS']
    
    target_names = [target]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target_names].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': 'D4',
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.3390/ma15082823",
        'items': None,
        'normalize': None,
    }


def read_yuan(target='CS', dataset=None, test_size=None, seed=None, plot=False):
    """
    Load and process Yuan (2022) recycled aggregate concrete dataset.

    Parameters:
    - target (str): Target column (e.g., 'CS' or 'FS').
    - plot (bool): If True, generate correlation heatmap.

    Returns: Dataset dictionary.
    """
    fn = './data/data_yuan/yuan2022_recycled_aggregate_concrete.txt'
    X = np.loadtxt(fn).reshape(-1, 14)
    cols = 'weffc acr rcar pcs nmrcas nmnas bdrca bdna warca wana larca lana CS FS'.split()
    X = pd.DataFrame(X, columns=cols)
    
    target_names = [target]
    if target == 'FS':
        X = X[X['FS'] != 0]
    
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target_names].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    if plot:
        df = X.copy()
        pl.figure(figsize=(5, 4))
        corr = df.corr().round(2)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
        pl.savefig(dataset + '_heatmap_correlation.png', bbox_inches='tight', dpi=300)
        pl.show()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': dataset,
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://www.mdpi.com/1996-1944/15/8/2823",
        'items': None,
        'normalize': None,
    }


def read_ngo(target='UCS', dataset=None, test_size=None, seed=None):
    """
    Load and process Ngo dataset.

    Parameters: Same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_ngo/ngo.txt'
    X = np.loadtxt(fn).reshape(-1, 16)
    cols = 'No D We Cc Cp S Mc T Ac Di L A V M De qu'.split()
    X = pd.DataFrame(X, columns=cols)
    X['UCS'] = X['qu'] / 1e3
    X.drop(['qu', 'No', 'L'], axis=1, inplace=True)
    
    target_names = [target]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target_names].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': dataset,
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.3390/app11041949",
        'items': None,
        'normalize': None,
    }


def read_jalal(target='UCS', dataset=None, test_size=None, seed=None):
    """
    Load and process Jalal dataset for soil compaction properties.

    Parameters: Same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_jalal/jalal.txt'
    X = np.loadtxt(fn).reshape(-1, 8)
    cols = 'CF wL wP IP Gs S pdmax wopt'.split()
    X = pd.DataFrame(X, columns=cols)
    
    target_names = ['pdmax', 'wopt']
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    n = 137  # Fixed split point from original code
    X_train, y_train = X[variable_names][:n].values, X[target_names][:n].values
    X_test, y_test = X[variable_names][n:].values, X[target_names][n:].values
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = pd.DataFrame(y_train, columns=target_names)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': dataset,
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.T,
        'X_test': X_test,
        'y_test': y_test.T,
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.1016/j.trgeo.2021.100608",
        'items': None,
        'normalize': None,
    }


def read_taffese(target='UCS', dataset=None, test_size=None, seed=None):
    """
    Load and process Taffese dataset.

    Parameters: Same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_taffese/taffese.txt'
    X = pd.read_csv(fn, header=None).values.ravel().reshape(-1, 10)
    cols = 'Soil Cement Lime LL PL PI USCS MDD OMC UCS'.split()
    X = pd.DataFrame(X, columns=cols)
    for c in X.columns.drop('USCS'):
        X[c] = X[c].astype(float)
    
    target_names = [target]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    categorical_columns = ['USCS']
    for cc in categorical_columns:
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target_names].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': 'D6',
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.3390/app11167503",
        'items': None,
        'normalize': None,
    }


def read_mahmoodzadeh(target='UCS', treatment='Cement', dataset=None, test_size=None, seed=None):
    """
    Load and process Mahmoodzadeh dataset for rock properties.

    Parameters: Same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_mahmoodzadeh/mahmoodzadeh.csv'
    X = pd.read_csv(fn)
    X.columns = ['No.', 'n', 'SHR', 'Vp', 'Is(50)', 'UCS', 'Rock-type']
    X.drop(['No.', 'Rock-type'], axis=1, inplace=True)
    
    target_names = [target]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target_names].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': dataset,
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.1016/j.trgeo.2020.100499",
        'items': None,
        'normalize': None,
    }


def read_mozumder(target='UCS', treatment='Cement', dataset=None, test_size=None, seed=None):
    """
    Load and process Mozumder dataset.

    Parameters: Same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_mozumder/mozumder.csv'
    X = pd.read_csv(fn)
    X.columns = ['#', 'ST', 'LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', 'UCS']
    variable_names = list(X.columns.drop([target, '#', 'ST']))
    X = X[variable_names + [target]]
    X.dropna(inplace=True)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': 'D5',
        'feature_names': np.array(variable_names),
        'target_names': [target],
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': [target],
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.1016/j.compgeo.2015.05.021",
        'items': None,
        'normalize': None,
    }


def read_tabarsa(target='UCS', treatment='Cement', dataset=None, test_size=None, seed=None):
    """
    Load and process Tabarsa dataset.

    Parameters: Same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_tabarsa/tabarsa.csv'
    X = pd.read_csv(fn)
    X.columns = ['#', 'ST', 'DUW', 'CT', 'C', 'L', 'RHA', 'UCS']
    variable_names = list(X.columns.drop([target, '#']))
    X = X[variable_names + [target]]
    X.dropna(inplace=True)
    
    categorical_columns = ['ST']
    for cc in categorical_columns:
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression'
    return {
        'task': task,
        'name': dataset,
        'feature_names': np.array(variable_names),
        'target_names': [target],
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': [target],
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "http://doi.org/10.1007/s11709-021-0689-9",
        'items': None,
        'normalize': None,
    }


def read_gajurel(target='UCS', treatment='Lime', dataset=None, test_size=None, seed=None):
    """
    Load and process Gajurel dataset for soil stabilization (Lime or Cement).

    Parameters:
    - treatment (str): 'Lime' or 'Cement'.
    - Other parameters same as read_burroughs.

    Returns: Dataset dictionary.
    """
    fn = './data/data_gajurel/US Soil Stabilization Database - Lime and Cement-1.xlsx'
    X = pd.read_excel(fn)
    X['OC'] = X['Organic.Content']
    X['UCS'] = X['UCS.psi'] / 145.038  # Convert psi to MPa
    X['USCS'] = X['Classification (USCS)']
    
    target_names = [target]
    classes = []
    if target != 'UCS':
        le = LabelEncoder()
        le.fit(X[target_names].values.ravel())
        X[target_names] = le.transform(X[target_names].values.ravel()).reshape(-1, 1)
        classes = dict(zip(le.transform(le.classes_), le.classes_))
    
    variable_names = ['LL', 'PL', 'PI', 'Clay', 'Silt', 'Sand', 'OC', treatment]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    
    if test_size == 0 or test_size is None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = pd.DataFrame([]).values, pd.DataFrame([]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target_names].values, test_size=test_size, shuffle=True, random_state=seed
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()
    
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f", index=True, 
                        caption='Basic statistics for dataset ' + dataset + '.')
    
    task = 'regression' if target == 'UCS' else 'classification'
    ds_name = dataset if dataset else target + ' ' + treatment
    return {
        'task': task,
        'name': ds_name,
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': target_names,
        'true_labels': classes,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://doi.org/10.1016/j.trgeo.2020.100506",
        'items': None,
        'normalize': None,
    }


def lhsu(xmin, xmax, nsample):
    """
    Generate Latin Hypercube Samples.

    Parameters:
    - xmin (array-like): Lower bounds for each variable.
    - xmax (array-like): Upper bounds for each variable.
    - nsample (int): Number of samples.

    Returns:
    - ndarray: Samples (nsample x nvar).
    """
    nvar = len(xmin)
    ran = np.random.rand(nsample, nvar)
    s = np.zeros((nsample, nvar))
    for j in range(nvar):
        idx = np.random.permutation(nsample)
        P = (idx - ran[:, j]) / nsample
        s[:, j] = xmin[j] + P * (xmax[j] - xmin[j])
    return s


if __name__ == "__main__":
    ds_names = ['D' + str(i + 1) for i in range(10)]
    for d in ds_names:
        D = read_ucs(d)
        print(D['name'], D['n_samples'])
        print(D['feature_names'])