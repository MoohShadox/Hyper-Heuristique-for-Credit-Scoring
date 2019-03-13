import abc
from itertools import combinations

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