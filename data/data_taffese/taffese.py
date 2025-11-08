import numpy as np, pandas as pd

fn='taffese.txt'
X = pd.read_csv(fn, header=None)
X = X.values.ravel().reshape(-1,10)
cols='Soil Cement Lime LL PL PI USCS MDD OMC UCS'
cols = cols.split(' ')
X = pd.DataFrame(X, columns=cols)
for c in X.columns.drop('USCS'):
    X[c] = X[c].astype(float)
        
    