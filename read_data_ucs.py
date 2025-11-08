# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split

# import matplotlib as mpl
# mpl.rcParams["text.usetex"] = False

np.bool = np.bool_

pl.rc("text", usetex=True)
pl.rc("font", **{"family": "serif", "serif": ["Palatino"]})

from sklearn.preprocessing import LabelEncoder


def read_ucs(dataset, test_size=None, seed=None):

    if dataset == "D1":
        return read_gajurel(
            target="UCS",
            treatment="Lime",
            dataset=dataset,
            test_size=test_size,
            seed=seed,
        )
    if dataset == "D2":
        return read_gajurel(
            target="UCS",
            treatment="Cement",
            dataset=dataset,
            test_size=test_size,
            seed=seed,
        )
    if dataset == "D3":
        return read_ngo(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "D4":
        return read_priyadarshee(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "D5":
        return read_mozumder(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "D6":
        return read_taffese(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "D7":
        return read_tabarsa(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "D8":
        return read_mahmoodzadeh(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "D9":
        return read_wang(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "D10":
        return read_zhang(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "RCA":
        return read_yuan(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "D0S":
        return read_burroughs(dataset=dataset, test_size=test_size, seed=seed)
    if dataset == "S1":
        return read_jalal(dataset=dataset, test_size=test_size, seed=seed)


# %%
def read_burroughs(target="UCS", dataset=None, test_size=None, seed=None):
    # %%
    fn = "./data/data_burroughs/burroughs.txt"
    fn = "./data/data_burroughs/whole.txt"
    X = pd.read_csv(fn, header=None)
    X = X.values.ravel()
    X = X.reshape(-1, 26)
    cols = "d50 Cu e0 OD600 Mu MCa FCa UCS".split(" ")
    X = pd.DataFrame(X, columns=cols)

    cols = [
        "Site",
        "Sample",
        "Number",
        "Determination",
        "Number",
        "L.L. (%)",
        "L.L. Class",
        "P.L. (%)",
        "P.L. Class",
        "P.I. (%)",
        "P.I. Class",
        "L.S. (%)",
        "L.S. (Class)",
        "Clay (%)",
        "Clay Class",
        "Sand (%)",
        "Sand Class",
        "Gravel (%)",
        "Gravel Class",
        "Moisture (%)",
        "Moisture Class",
        "Density (t/m3)",
        "Density Code",
        "UCS (MPa)",
        "UCS Code",
        "Lime (%)",
        "Cement (%)," "Asphalt (%)",
    ]

    # median grain size , coefficient of uniformity , initial void ratio , optical density of bacterial suspension , urea concentration , calcium concentration  and calcium carbonate content . Based on this criterion, totally 351 bio-cemented sand samples were collected from the literature

    for c in X.columns:  # .drop('UCS'):
        X[c] = X[c].astype(float)

    # X.drop(['MDD', 'OMC'], axis=1, inplace=True)
    target_names = ["UCS"]

    variable_names = list(X.columns.drop(target_names))
    # variable_names = ['LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', ]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = []
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    X_train, y_train = X[variable_names], X[target_names]
    X_test, y_test = pd.DataFrame(
        [
            [],
        ]
    ), pd.DataFrame(
        [
            [],
        ]
    )

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    # pl.figure(figsize=(5, 4))
    # corr = df.corr().round(2)
    # mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    ##heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    # heatmap.set_title(dataset+': Correlation Heatmap ', fontdict={'fontsize':12}, pad=12);
    # pl.savefig(dataset+'_heatmap_correlation'+'.png',  bbox_inches='tight', dpi=300)
    # pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = X_train.copy()
    df_train[target_names] = y_train
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression"  # if target_names[0]=='UCS' else 'classification'
    regression_data = {
        "task": task,
        "name": dataset,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train.values,
        "y_train": y_train.values.T,
        "X_test": X_test.values,
        "y_test": [[]] * len(target_names),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.1016/j.jclepro.2021.128205",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_wang(target="UCS", dataset=None, test_size=None, seed=None):
    # %%
    fn = "./data/data_wang/wang2021.txt"
    X = np.loadtxt(fn)
    X = X.reshape(-1, 8)
    cols = "d50 Cu e0 OD600 Mu MCa FCa UCS".split(" ")
    X = pd.DataFrame(X, columns=cols)

    # median grain size , coefficient of uniformity , initial void ratio , optical density of bacterial suspension , urea concentration , calcium concentration  and calcium carbonate content . Based on this criterion, totally 351 bio-cemented sand samples were collected from the literature

    for c in X.columns:  # .drop('UCS'):
        X[c] = X[c].astype(float)

    # X.drop(['MDD', 'OMC'], axis=1, inplace=True)
    target_names = ["UCS"]

    variable_names = list(X.columns.drop(target_names))
    # variable_names = ['LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', ]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = []
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    pl.figure(figsize=(5, 4))
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
    # heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title(
        dataset + ": Correlation Heatmap ", fontdict={"fontsize": 12}, pad=12
    )
    pl.savefig(dataset + "_heatmap_correlation" + ".png", bbox_inches="tight", dpi=300)
    pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression"  # if target_names[0]=='UCS' else 'classification'
    regression_data = {
        "task": task,
        "name": dataset,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.1016/j.jclepro.2021.128205",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_zhang(target="UCS", dataset=None, test_size=None, seed=None):
    # %%
    fn = "./data/data_zhang/zhang2022.csv"
    X = pd.read_csv(fn, header=0)
    X.drop(["Specimen"], axis=1, inplace=True)
    for c in X.columns:  # .drop('UCS'):
        X[c] = X[c].astype(float)

    # X.drop(['MDD', 'OMC'], axis=1, inplace=True)
    target_names = ["UCS", "Tp", "Tt "]
    target_names = [
        "UCS",
    ]

    variable_names = list(X.columns.drop(target_names))
    # variable_names = ['LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', ]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = []
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    pl.figure(figsize=(5, 4))
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
    # heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title(
        dataset + ": Correlation Heatmap ", fontdict={"fontsize": 12}, pad=12
    )
    pl.savefig(dataset + "_heatmap_correlation" + ".png", bbox_inches="tight", dpi=300)
    pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression"  # if target_names[0]=='UCS' else 'classification'
    regression_data = {
        "task": task,
        "name": dataset,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.1016/j.jmrt.2022.02.076",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_priyadarshee(target="UCS", dataset=None, test_size=None, seed=None):
    # %%
    fn = "./data/data_priyadarshee/data_priyadarshee.csv"
    X = pd.read_csv(fn, header=0)
    X.drop(["Number", "Predicted"], axis=1, inplace=True)
    for c in X.columns.drop("UCS"):
        X[c] = X[c].astype(float)

    X.columns = ["Clay", "Pond Ash", "Rice Husk", "Cement", "Curing", "UCS"]
    X.columns = ["C", "PondAsh", "RiceHusk", "Cement", "Curing", "UCS"]

    # X.drop(['MDD', 'OMC'], axis=1, inplace=True)
    target = "UCS"
    target_names = [target]

    variable_names = list(X.columns.drop(target_names))
    # variable_names = ['LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', ]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = []
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    pl.figure(figsize=(5, 4))
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
    # heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title(
        dataset + ": Correlation Heatmap ", fontdict={"fontsize": 12}, pad=12
    )
    pl.savefig(dataset + "_heatmap_correlation" + ".png", bbox_inches="tight", dpi=300)
    pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression" if target_names[0] == "UCS" else "classification"
    regression_data = {
        "task": task,
        "name": "D4",
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.3390/ma15082823",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_yuan(
    target="CS",
    dataset=None,
    test_size=None,
    seed=None,
    plot=False,
):
    # %%
    fn = "./data/data_yuan/yuan2022_recycled_aggregate_concrete.txt"
    X = np.loadtxt(fn)
    X = X.reshape(-1, 14)
    cols = (
        "weffc acr rcar pcs nmrcas nmnas bdrca bdna warca wana larca lana CS FS".split(
            " "
        )
    )
    X = pd.DataFrame(X, columns=cols)

    target_names = [target]

    if target == "FS":
        X = X[X["FS"] != 0]

    variable_names = list(X.columns.drop(target_names))
    # variable_names = ['LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', ]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = []
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    # if plot:
    # pl.figure(figsize=(5, 4))
    # corr = df.corr().round(2)
    # mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    ##heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    # heatmap.set_title(dataset+': Correlation Heatmap ', fontdict={'fontsize':12}, pad=12);
    # pl.savefig(dataset+'_heatmap_correlation'+'.png',  bbox_inches='tight', dpi=300)
    # pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression"  # if target_names[0]=='UCS' else 'classification'
    regression_data = {
        "task": task,
        "name": dataset,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://www.mdpi.com/1996-1944/15/8/2823",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_ngo(target="UCS", dataset=None, test_size=None, seed=None):
    # %%
    fn = "./data/data_ngo/ngo.txt"
    X = np.loadtxt(fn)
    X = X.reshape(-1, 16)
    cols = "No D We Cc Cp S Mc T Ac Di L A V M De qu".split(" ")
    # cols = 'No D We Cc Cp S Mc T Ac Di L A V MS De qu'.split(' ')
    X = pd.DataFrame(X, columns=cols)
    X["UCS"] = X["qu"] / 1e3
    X.drop(["qu", "No", "L"], axis=1, inplace=True)

    target = "UCS"
    target_names = [target]

    variable_names = list(X.columns.drop(target_names))
    # variable_names = ['LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', ]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = []
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    # X_train, y_train = X[variable_names], X[target_names]
    # X_test , y_test  = pd.DataFrame([[],]), pd.DataFrame([[],])

    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    pl.figure(figsize=(7, 6))
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
    # heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title(
        dataset + ": Correlation Heatmap ", fontdict={"fontsize": 12}, pad=12
    )
    pl.yticks(rotation=0)
    pl.savefig(dataset + "_heatmap_correlation" + ".png", bbox_inches="tight", dpi=300)
    pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    # df_train=X_train.copy(); df_train[target_names]=y_train
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression" if target_names[0] == "UCS" else "classification"
    regression_data = {
        "task": task,
        "name": dataset,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.3390/app11041949",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_jalal(target="UCS", dataset=None, test_size=None, seed=None):
    # %%
    fn = "./data/data_jalal/jalal.txt"
    X = np.loadtxt(fn)
    X = X.reshape(-1, 8)
    cols = "CF wL wP IP Gs S pdmax wopt".split(" ")
    X = pd.DataFrame(X, columns=cols)

    target_names = ["pdmax", "wopt"]

    variable_names = list(X.columns.drop(target_names))
    # variable_names = ['LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', ]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = []
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    n = 137
    X_train, y_train = X[variable_names][:n], X[target_names][:n]
    X_test, y_test = X[variable_names][n:], X[target_names][n:]

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    # pl.figure(figsize=(5, 4))
    # corr = df.corr().round(2)
    # mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    ##heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    # heatmap.set_title(dataset+': Correlation Heatmap ', fontdict={'fontsize':12}, pad=12);
    # pl.savefig(dataset+'_heatmap_correlation'+'.png',  bbox_inches='tight', dpi=300)
    # pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = X_train.copy()
    df_train[target_names] = y_train
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression"  # if target_names[0]=='UCS' else 'classification'
    regression_data = {
        "task": task,
        "name": dataset,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train.values,
        "y_train": y_train.values.T,
        "X_test": X_test.values,
        "y_test": y_test.values.T,
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.1016/j.trgeo.2021.100608",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%


# %%
def read_taffese(target="UCS", dataset=None, test_size=None, seed=None):
    # %%
    fn = "./data/data_taffese/taffese.txt"
    X = pd.read_csv(fn, header=None)
    X = X.values.ravel().reshape(-1, 10)
    cols = "Soil Cement Lime LL PL PI USCS MDD OMC UCS"
    cols = cols.split(" ")
    X = pd.DataFrame(X, columns=cols)
    for c in X.columns.drop("USCS"):
        X[c] = X[c].astype(float)

    # X.drop(['MDD', 'OMC'], axis=1, inplace=True)
    target = "UCS"
    target_names = [target]

    variable_names = list(X.columns.drop(target_names))
    # variable_names = ['LL', 'PI', 'S', 'FA', 'M', 'A/B', 'Na/Al', 'Si/Al', ]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = ["USCS"]
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    pl.figure(figsize=(5, 4))
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
    # heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title(
        dataset + ": Correlation Heatmap ", fontdict={"fontsize": 12}, pad=12
    )
    pl.savefig(dataset + "_heatmap_correlation" + ".png", bbox_inches="tight", dpi=300)
    pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression" if target_names[0] == "UCS" else "classification"
    regression_data = {
        "task": task,
        "name": "D6",  # target_names[0]+' '+treatment,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.3390/app11167503",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_mahmoodzadeh(
    target="UCS", treatment="Cement", dataset=None, test_size=None, seed=None
):
    # %%
    fn = "./data/data_mahmoodzadeh/mahmoodzadeh.csv"
    X = pd.read_csv(fn)
    X.columns = ["No.", "n", "SHR", "Vp", "Is(50)", "UCS", "Rock-type"]
    X.drop(
        [
            "No.",
            "Rock-type",
        ],
        axis=1,
        inplace=True,
    )
    target = "UCS"
    target_names = [target]

    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = []
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    pl.figure(figsize=(5, 4))
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
    # heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title(
        dataset + ": Correlation Heatmap ", fontdict={"fontsize": 12}, pad=12
    )
    pl.savefig(dataset + "_heatmap_correlation" + ".png", bbox_inches="tight", dpi=300)
    pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression" if target_names[0] == "UCS" else "classification"
    regression_data = {
        "task": task,
        "name": dataset,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.1016/j.trgeo.2020.100499",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_mozumder(
    target="UCS", treatment="Cement", dataset=None, test_size=None, seed=None
):
    # %%
    fn = "./data/data_mozumder/mozumder.csv"
    X = pd.read_csv(fn)
    target = "UCS"
    target_names = [target]

    X.columns = ["#", "ST", "LL", "PI", "S", "FA", "M", "A/B", "Na/Al", "Si/Al", "UCS"]
    # X.columns = ['#', 'ST', 'LL', 'PI', 'GGBS', 'FA', 'MC', 'A/B', 'Na/Al', 'Si/Al', 'UCS']
    variable_names = list(X.columns.drop(target_names + ["#", "ST"]))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = []
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    pl.figure(figsize=(5, 4))
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
    # heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title(
        dataset + ": Correlation Heatmap ", fontdict={"fontsize": 12}, pad=12
    )
    pl.savefig(dataset + "_heatmap_correlation" + ".png", bbox_inches="tight", dpi=300)
    pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression" if target_names[0] == "UCS" else "classification"
    regression_data = {
        "task": task,
        "name": "D5",  # target_names[0]+' '+treatment,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.1016/j.compgeo.2015.05.021",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_tabarsa(
    target="UCS", treatment="Cement", dataset=None, test_size=None, seed=None
):
    # %%
    fn = "./data/data_tabarsa/tabarsa.csv"
    X = pd.read_csv(fn)
    X.columns = ["#", "ST", "DUW", "CT", "C", "L", "RHA", "UCS"]
    target = "UCS"
    target_names = [target]

    variable_names = list(X.columns.drop(target_names + ["#"]))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    categorical_columns = ["ST"]
    for cc in categorical_columns:
        # print(cc)
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)
        # classes = dict(zip(le.transform(le.classes_), le.classes_))

    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    pl.figure(figsize=(5, 4))
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
    # heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title(
        dataset + ": Correlation Heatmap ", fontdict={"fontsize": 12}, pad=12
    )
    pl.savefig(dataset + "_heatmap_correlation" + ".png", bbox_inches="tight", dpi=300)
    pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression" if target_names[0] == "UCS" else "classification"
    regression_data = {
        "task": task,
        "name": dataset,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "http://doi.org/10.1007/s11709-021-0689-9",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_kardani(
    target="UCS", treatment="Cement", dataset=None, test_size=None, seed=None
):
    # %%
    fn = "./data/data_kardani/kardani.csv"
    X = pd.read_csv(fn)
    X.columns = ["CC", "S", "DD", "UCS"]
    # X.columns = ['Cement', 'Suction', 'Dry Density', 'UCS']
    target = "UCS"
    target_names = [target]

    variable_names = list(X.columns.drop(target_names))

    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    X_train, y_train = X[variable_names], X[target_names]
    X_test, y_test = pd.DataFrame(
        [
            [],
        ]
    ), pd.DataFrame(
        [
            [],
        ]
    )

    # if test_size==0:
    #    X_train = pd.concat([X_train, X_test])
    #    y_train = pd.concat([y_train, y_test])
    #    X_test = pd.DataFrame([],columns=X_test.columns)
    #    y_test = pd.DataFrame([],columns=y_test.columns)

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    # pl.figure(figsize=(5, 4))
    # corr = df.corr().round(2)
    # mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    ##heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    # heatmap.set_title(dataset+': Correlation Heatmap ', fontdict={'fontsize':12}, pad=12);
    # pl.savefig(dataset+'_heatmap_correlation'+'.png',  bbox_inches='tight', dpi=300)
    # pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    df_train = X_train.copy()
    df_train[target_names] = y_train
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    task = "regression" if target_names[0] == "UCS" else "classification"
    regression_data = {
        "task": task,
        "name": "D3",  # target_names[0]+' '+treatment,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train.values,
        "y_train": y_train.values.T,
        "X_test": X_test.values,
        "y_test": [[]],
        "targets": target_names,
        #'true_labels'     : classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.1016/j.trgeo.2021.100591",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def read_gajurel(
    target="UCS", treatment="Lime", dataset=None, test_size=None, seed=None
):
    # %%
    fn = "./data/data_gajurel/US Soil Stabilization Database - Lime and Cement-1.xlsx"
    X = pd.read_excel(fn)
    # X.drop('n', axis=1, inplace=True)
    # U = pd.read_csv('./data/data_ggbs/ggbs_test.csv', sep=';', header=0)
    # U.drop(['GEP2','n'], axis=1, inplace=True)

    # for i in range(len(X)):
    #    for j in range(len(U)):
    #        if np.linalg.norm(X.iloc[i].values - U.iloc[j].values)==0:
    #            X.iloc[i]=None

    # X.dropna(inplace=True)

    # X.dropna(inplace=True)
    # X.to_latex(buf='ggbs_train'+'.tex', index=False)
    # U.to_latex(buf='ggbs_test'+'.tex', index=False)
    # X.columns = [x.replace(' [%]','') for x in X.columns]

    # X.groupby('AS (day)').describe().T
    # U.groupby('AS (day)').describe().T

    # treatment='Lime'
    # treatment='Cement'

    # target_names=['UCS.psi']
    # variable_names = X.columns.drop(target_names)
    X["OC"] = X["Organic.Content"]
    X["UCS"] = X["UCS.psi"]
    X["USCS"] = X["Classification (USCS)"]
    X["UCS"] /= 145.038  # convert to MPa
    classes = []

    if target == "UCS":
        target_names = ["UCS"]
    else:
        target_names = ["USCS"]
        le = LabelEncoder()
        le.fit(X[target_names].values.ravel())
        X[target_names] = le.transform(X[target_names].values.ravel()).reshape(-1, 1)
        classes = dict(zip(le.transform(le.classes_), le.classes_))

    variable_names = [
        "LL",
        "PL",
        "PI",
        "Clay",
        "Silt",
        "Sand",
        "OC",
    ] + [treatment]
    X = X[variable_names + target_names]
    X.dropna(inplace=True)
    if test_size == 0 or test_size == None:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = (
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
            pd.DataFrame(
                [
                    [],
                ]
            ).values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values,
            X[target_names].values,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
        )
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()
    # df.columns = [x.replace('(wt%)','') for x in df.columns]

    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    # print(stat_train.to_latex(),)
    stat_train.to_latex(
        buf=(dataset + "_train" + ".tex").lower(),
        float_format="%.2f",
        index=True,
        caption="Basic statistics for dataset " + dataset + ".",
    )

    pl.figure(figsize=(5, 4))
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
    # heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title(
        dataset + ": Correlation Heatmap ", fontdict={"fontsize": 12}, pad=12
    )
    pl.yticks(rotation=0)
    pl.savefig(dataset + "_heatmap_correlation" + ".png", bbox_inches="tight", dpi=300)
    pl.show()

    n = len(y_train)
    n_samples, n_features = X_train.shape

    task = "regression" if target_names[0] == "UCS" else "classification"

    if dataset == None:
        ds_name = target_names[0] + " " + treatment
    else:
        ds_name = dataset

    regression_data = {
        "task": task,
        "name": ds_name,
        "feature_names": np.array(variable_names),
        "target_names": target_names,
        "n_samples": n_samples,
        "n_features": n_features,
        "X_train": X_train,
        "y_train": y_train.reshape(1, -1),
        "X_test": X_test,
        "y_test": y_test.reshape(1, -1),
        "targets": target_names,
        "true_labels": classes,
        "predicted_labels": None,
        "descriptions": "None",
        "reference": "https://doi.org/10.1016/j.trgeo.2020.100506",
        "items": None,
        "normalize": None,
    }
    # %%
    return regression_data


# %%
def lhsu(xmin, xmax, nsample):
    nvar = len(xmin)
    ran = np.random.rand(nsample, nvar)
    s = np.zeros((nsample, nvar))
    for j in range(nvar):
        idx = np.random.permutation(nsample)
        P = (idx.T - ran[:, j]) / nsample
        s[:, j] = xmin[j] + P * (xmax[j] - xmin[j])

    return s


if __name__ == "__main__":

    # D = read_gajurel(target='UCS', treatment='Lime')
    # D = read_gajurel(target='UCS', treatment='Cement')
    # D = read_kardani()
    ds_names = ["D" + str(i + 1) for i in range(10)]
    ds_names += [
        "S1",
        "RCA",
    ]
    for d in ds_names:
        D = read_ucs(d)
        print(D["name"], D["n_samples"])
        print(D["feature_names"])
