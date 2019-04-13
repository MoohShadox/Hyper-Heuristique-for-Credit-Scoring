import requests
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Destiny.Evaluateur_Precision import Evaluateur_Precision



def load_promoter_dataset():
    r = open (r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\promoters.data.txt")
    L = r.readlines()
    X = []
    for i in L:
        K = np.array (i.strip().replace(r"\t\t","").split(","))
        X.append (K)
    X = np.array(X)
    X = X.transpose()
    Y = []
    for i in X[0]:
        if(i=="+"):
            Y.append(1)
        else:
            Y.append(0)
    R = []
    for i in X[2]:
        gen = []
        for g in i:
            if(g == 'a'):
                gen.append(np.array(1).astype("float64"))
            elif(g == 'c'):
                gen.append(np.array(2).astype("float64"))
            elif(g == 't'):
                gen.append(np.array(3).astype("float64"))
            elif(g=='g'):
                gen.append(np.array(4).astype("float64"))
        R.append(gen)
    return np.array(R),np.array(Y)

train,target = load_promoter_dataset()
print(train.shape)
print(target.shape)
#E = Evaluateur_Precision(train,target.ravel())
#E.train(KNeighborsClassifier())
#print(E.vecteur_precision())