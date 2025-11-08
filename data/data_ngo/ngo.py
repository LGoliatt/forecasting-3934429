import numpy as np, pandas as pd

fn='ngo.txt'
X = np.loadtxt(fn)
X = X.reshape(-1,16)
cols = 'No D We Cc Cp S Mc T Ac Di L A V M De qu'.split(' ')
X = pd.DataFrame(X, columns=cols)
X['UCS']=X['qu']/1e3
X.drop('qu', axis=1, inplace=True)
