import numpy as np
from sklearn.decomposition import PCA

class PCA2(object):
    def __init__(self):
        self.__feature_scores = None
        self.__vectors = None


    def fit(self, X,Y):
        P = PCA()
        P.fit(X,Y)
        #mean_vec = np.mean (X , axis=0)
        #cov_mat = (X - mean_vec).T.dot ((X - mean_vec)) / (X.shape[0] - 1)
        self.__feature_scores , self.__vectors = P.explained_variance_ratio_,P.components_



    def Score(self):
        print("Scores ", self.__feature_scores)
        L = len(self.__vectors[0])*[0]
        A = np.array(L)
        for i in range(0,len(self.__vectors)-1):
            I = np.array(self.__vectors[i])
            I = self.__feature_scores[i] * I
            A = A + I
        return A

from Destiny.DataSets import german_dataset
data, target = german_dataset.load_german_dataset()


