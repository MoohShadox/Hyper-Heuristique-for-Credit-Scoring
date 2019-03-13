import numpy as np
import pandas as pd

def load_german_dataset():
    X = pd.read_csv(r"C:\Users\Geekzone\Documents\PFEHyperHeuristicForCreditScoring\Destiny\DataSets\GermanData.csv")
    X = np.array(X)
    Y = X.transpose()[-1]
    Y = Y.astype("int")
    X = X.transpose()[:-1]
    X = X.transpose()
    W = []
    for i in X:
        L = []
        cpt = 1
        for j in i:
            l = j
            if (str(j).startswith("A"+str(cpt))):
                l = int(str(j).replace("A"+str(cpt),""))
            L.append(float(l))
            cpt = cpt + 1
        W.append(L)
    return np.array(W),Y

