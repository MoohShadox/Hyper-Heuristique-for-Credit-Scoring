import abc
from itertools import combinations
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class Mesure:

    def __init__(self):
        self._liste_mesures = []
        self._calculated_measures = {}
        self._attributs = {}
        self._target = {}


    def fit(self,data,target):
        d = data.transpose()
        cpt = 0
        for i in d:
            self._attributs[str(cpt)] = i
            cpt = cpt +1
        self._attributs["-1"] = target


    def ranking_function_constructor(self,motclef):
        pass


    def ranked_attributs(self,motclef,nb=1):
        if not motclef in self._liste_mesures:
            return None
        ranker = self.ranking_function_constructor(motclef)
        scores = []
        L = range(0,len(self._attributs.keys())-2)
        for i in combinations(L,nb):
            K = []
            for j in i:
                K.append(j)
            t = i,ranker(tuple(K))
            scores.append(t)
        X = []
        for i in scores:
            X.append(i[1])
        X = np.array(X)
        X = X.reshape(-1,1)
        sc = MinMaxScaler()
        X = sc.fit_transform(X)
        cpt = 0
        XX  = []
        X = X.transpose()
        X = list(X[0])
        #print("X = ",X)
        for i in scores:
            t = i[0],X[cpt]
            XX.append(t)
            cpt = cpt + 1
        #print("XX = " , XX)
        scores.sort(key=lambda x:x[1],reverse=True)
        return scores

    def getCalculatedMeasures(self):
        return self._calculated_measures

    def rank_with(self,lmotclef = None,n = 1):
        if not n in self._calculated_measures.keys():
            self._calculated_measures[n] = {}
        if(lmotclef == None):
            lmotclef = self._liste_mesures
        for mc in lmotclef:
            if not (mc in self._calculated_measures[n].keys()):
                self._calculated_measures[n][mc] = self.ranked_attributs(mc,nb=n)
        return self._calculated_measures