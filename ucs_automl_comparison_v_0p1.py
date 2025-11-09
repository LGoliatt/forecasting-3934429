#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings

# warnings.filterwarnings("ignore")
np.float = np.float64
import glob as gl
import pylab as pl
import os
import json
import time
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)  # , root_mean_squared_error

# from autosklearn.regression import AutoSklearnRegressor
from flaml import AutoML
from tpot import TPOTRegressor
from autokeras import StructuredDataRegressor

# import h2o
# from h2o.automl import H2OAutoML
# h2o.init()
from autogluon.tabular import TabularDataset, TabularPredictor

import h2o
from h2o.automl import H2OAutoML

h2o.init(nthreads=2)  # Adicionado multithreading por ROMULO MURUCCI
# h2o.init()

from read_data_ucs import *

np.bool = np.bool_
np.object = object
basename = "ucs_automl_"

# %%----------------------------------------------------------------------------
pd.options.display.float_format = "{:.3f}".format

time_budget = 120  # seconds
n_runs = 30
n_splits = 5
epochs = 100
scoring = "neg_root_mean_squared_error"
for run in range(1, n_runs + 1):
    # random_seed=1
    seed = random_seed = run * 37 + 1001
    test_size = 0.30
    datasets = [
        read_ucs("D1", test_size=test_size, seed=random_seed),
        read_ucs("D2", test_size=test_size, seed=random_seed),
        read_ucs("D3", test_size=test_size, seed=random_seed),
        read_ucs("D4", test_size=test_size, seed=random_seed),
        read_ucs("D5", test_size=test_size, seed=random_seed),
        read_ucs("D6", test_size=test_size, seed=random_seed),
        read_ucs("D7", test_size=test_size, seed=random_seed),
        read_ucs("D8", test_size=test_size, seed=random_seed),
        read_ucs("D9", test_size=test_size, seed=random_seed),
        read_ucs("D10", test_size=test_size, seed=random_seed),
    ]

    for dataset in datasets:  # [:1]:
        dr = dataset["name"].replace(" ", "_").replace("'", "").lower()
        path = "./json_automl_" + dr + "/"
        os.system("mkdir -p " + path.replace(" ", "_").replace("-", "_").lower())

        for tk, tn in enumerate(dataset["target_names"]):
            print(tk, tn)
            dataset_name = dataset["name"] + "-" + tn
            target = dataset["target_names"][tk]
            y_train, y_test = dataset["y_train"][tk], dataset["y_test"][tk]
            X_train, X_test = dataset["X_train"], dataset["X_test"]
            n_samples_train, n_features = dataset["n_samples"], dataset["n_features"]
            task, normalize = dataset["task"], dataset["normalize"]
            feature_names = dataset["feature_names"]
            n_samples_test = len(y_test)

            s = "" + "\n"
            s += "=" * 80 + "\n"
            s += "Dataset                    : " + dataset_name + " -- " + target + "\n"
            s += "Output                     : " + tn + "\n"
            s += "Number of training samples : " + str(n_samples_train) + "\n"
            s += "Number of testing  samples : " + str(n_samples_test) + "\n"
            s += "Number of features         : " + str(n_features) + "\n"
            s += "Normalization              : " + str(normalize) + "\n"
            s += "Task                       : " + str(dataset["task"]) + "\n"
            # s+='Reference                  : '+str(dataset['reference'])+'\n'
            s += "=" * 80
            s += "\n"

            scoring = (
                "f1_micro"
                if task == "classification"
                else "neg_root_mean_squared_error"
            )

            print(s)
            e = 1e-5

            train_data = pd.DataFrame(X_train, columns=feature_names)
            train_data[target] = y_train
            test_data = pd.DataFrame(X_test, columns=feature_names)
            test_data[target] = y_test

            train_df = h2o.H2OFrame(train_data)
            test_df = h2o.H2OFrame(test_data)

            # FLAML
            flaml = AutoML()
            flaml_settings = {
                "time_budget": time_budget,  # in seconds
                "metric": "mae",
                "task": "regression",
                "log_file_name": "ucs.log",
                "estimator_list": [
                    "lgbm",
                    "rf",
                    "xgboost",
                    "extra_tree",
                    "xgb_limitdepth",
                    # "sgd", ### sgd is not a built-in learner. Please use AutoML.add_learner() to add a customized learner.
                    "catboost",
                    "kneighbor",
                    # "histgb", ### histgb is not a built-in learner. Please use AutoML.add_learner() to add a customized learner.
                ],
                "seed": seed,
                "verbose": False,
                "n_jobs": 2,  # Adicionado multithreading por ROMULO MURUCCI
            }
            # TPOT
            pipeline_optimizer = TPOTRegressor(
                generations=20,
                population_size=20,
                cv=5,
                random_state=seed,
                verbosity=False,
                n_jobs=2,  # Adicionado multithreading por ROMULO MURUCCI
            )

            # AutoSklearn
            include = {
                "regressor": [  #'adaboost', 'ard_regression', 'decision_tree',
                    "extra_trees",
                    "gaussian_process",
                    "gradient_boosting",
                    "k_nearest_neighbors",
                    "liblinear_svr",
                    "libsvm_svr",
                    "mlp",
                    "random_forest",
                    "sgd",
                ],
                # "data_preprocessor": ["no_preprocessing"],
                "feature_preprocessor": ["no_preprocessing"],
            }

            for auto in [
                # "AutoSklearn",
                "AutoGluon",
                "AutoKeras",
                "H2O",
                "TPOT",
                "FLAML",
            ]:
                start_time = time.time()
                print(auto)
                if auto == "FLAML":
                    automl = flaml
                    automl.fit(X_train=X_train, y_train=y_train, **flaml_settings)
                    y_pred = automl.predict(X_test)
                elif auto == "TPOT":
                    automl = pipeline_optimizer
                    automl.fit(X_train, y_train)
                    y_pred = automl.predict(X_test)
                elif auto == "AutoGluon":
                    automl = TabularPredictor(label=target).fit(
                        train_data=train_data,
                        # eval_metric='mean_absolute_error',
                        verbosity=False,
                        # num_cpus=4,  # O modelo já avalia o multithreading automatico
                    )
                    y_pred = automl.predict(test_data).values
                    # REMOVIDO "elif auto == "AutoSklearn":" ################################################
                elif auto == "AutoKeras":
                    # X_train = X_train.astype('float32')
                    # y_train = y_train.astype('float32')
                    automl = StructuredDataRegressor(
                        max_trials=50,
                        column_names=list(feature_names),
                        # overwrite=True,
                        loss="mean_absolute_error",
                        seed=seed,
                    )
                    automl.fit(x=X_train, y=y_train, epochs=epochs, verbose=False)
                    y_pred = automl.predict(X_test).ravel()
                elif auto == "H2O":
                    automl = H2OAutoML(
                        max_runtime_secs=time_budget,
                        # qexclude_algos =['DeepLearning'],
                        seed=seed,
                        # stopping_metric ='logloss',
                        sort_metric="rmse",
                        # balance_classes = False,
                    )
                    automl.train(
                        x=list(feature_names), y=target, training_frame=train_df
                    )
                    y_pred = automl.predict(test_df).as_data_frame().values.ravel()
                else:
                    pass

                elapsed_time = time.time() - start_time
                ds_name, est_name, tg_name = dataset_name, auto, target
                # print('='*60+'\n'+str(run)+' - '+str(seed)+'\n'+'='*60)
                # pl.plot(y_test,y_test,'k-', y_test, y_pred,'ro')
                # pl.title(ds_name+' - '+tg_name+': '+auto+' - '+str(r2_score(y_test,y_pred)))
                # pl.gca().set_aspect('equal')
                # pl.xlabel('Observed values')
                # pl.ylabel('Predicted values')
                # pl.show()
                print(">> ", run, auto, "\t\t", r2_score(y_test, y_pred), elapsed_time)

                X = []
                l = {}

                l["run"] = run
                l["elapsed_time"] = elapsed_time
                l["seed"] = seed
                l["estimator"] = auto
                l["y_pred"] = y_pred.tolist()
                l["y_test"] = y_test.tolist()
                l["r2"] = r2_score(y_test, y_pred)
                l["mae"] = mean_absolute_error(y_test, y_pred)
                l["mse"] = mean_squared_error(y_test, y_pred)
                l["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
                l["dataset"] = dataset_name
                l["target"] = target

                X.append(l)

                pk = (
                    path  #'_'+
                    + basename
                    + "_"
                    + "_run_"
                    + str("{:02d}".format(run))
                    + "_"
                    + ("%15s" % ds_name).rjust(15).replace(" ", "_")  #'_'+
                    + ("%11s" % est_name).rjust(12).replace(" ", "_")  #'_'+
                    +
                    # ("%10s"%alg_name        ).rjust(10).replace(' ','_')+#'_'+
                    ("%15s" % tg_name).rjust(16).replace(" ", "_")  #'_'+
                    +
                    # ("%15s"%os.uname()[1]   ).rjust(25).replace(' ','_')+#'_'+
                    # time.strftime("%Y_%m_%d_") + time.strftime("_%Hh_%Mm_%S")+
                    ".json"
                )
                pk = pk.replace(" ", "_").replace("'", "").lower()
                pk = pk.replace("(", "_").replace(")", "_").lower()
                pk = pk.replace("[", "_").replace("]", "_").lower()
                pk = pk.replace("-", "_").replace("_", "_").lower()
                pk = pk.replace("{", "").replace("}", "").lower()
                pk = pk.replace("$", "").replace("}", "").lower()

                with open(pk, "w") as fp:
                    json.dump(X, fp)

# %%----------------------------------------------------------------------------
