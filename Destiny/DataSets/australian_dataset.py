import numpy as np
import requests

def load_australian_dataset():
    print('h')
    r = requests.get(r"http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat")
    L = str(r.content).split(r'\n')
    X = []
    for i in L:
        K = np.array(i.split(' '))
        X.append(K)
    D = np.array(X[:-1])
    D = D.transpose()
    Y = D[-1]
    print(Y)
    X = np.array(X)[:-2]
    print(X)
    return X,Y

def save_dataset_on_disc():
    try:
        with open("australian_dataset.txt","w") as f:
            r = requests.get (r"http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat")
            f.write(str(r.content).replace("\n","\n"))
            f.close()
    except(Exception):
        print("retry")
        save_dataset_on_disc()



save_dataset_on_disc()