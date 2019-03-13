import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA(object):
    def __init__(self, n_neighbors=100, n_features_to_keep=10):
        self.__variance_ratio = None
        self.__components = None


    def fit(self, X,Y):
        LD = LinearDiscriminantAnalysis()
        from sklearn.preprocessing import StandardScaler
        St = StandardScaler()
        X = St.fit_transform(X)
        LD.fit(X,Y)
        self.__components = LD.coef_
        self.__scores = []
        cpt = 0
        for i in LD.coef_[0]:
            self.__scores.append((cpt,i))
            cpt = cpt + 1


    def getScores(self):
        return self.__scores


