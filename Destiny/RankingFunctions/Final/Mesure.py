import abc
from itertools import combinations
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from Destiny import Tresholding
from Destiny.Embedded_Thresholding import Embedded_Thresholding


class Mesure:

    #Taille de Méga Attribut :
    MegaAttributTailleMax = 3

    def __init__(self):
        self._liste_mesures = []
        self._liste_thresholds = []
        self._calculated_measures = {}
        self._attributs = {}
        self._subsets = None
        self._target = {}
        self.__data = None
        self.__target = None


    def CreateSubsets(self,borne=None):
        self._subsets = {}
        T = Embedded_Thresholding()
        T.fit(self.__data,self.__target)
        for i in range(2,Mesure.MegaAttributTailleMax+1):
            self._subsets[i] = T.generer_subset(i , borne)


    def setThresholdsAutomatiquement(self,s=None):
        self.rank_with(n=1)
        T = Tresholding.Tresholding()
        if(s==None):
            T.fit(self.__data,self.__target)
            s = T.getTreshold(self.__data,self.__target)
        for j in self._calculated_measures[1]:
            if(self._liste_thresholds[self._liste_mesures.index (j)]==0):
                self._liste_thresholds[self._liste_mesures.index(j)] = self._calculated_measures[1][j][int(s*(len(self._attributs.keys())-1))][1]
        self._calculated_measures.clear()


    def fit(self,data,target):
        self.__data = data
        self.__target = target
        d = data.transpose()
        cpt = 0
        for i in d:
            self._attributs[str(cpt)] = i
            cpt = cpt +1
        self._attributs["-1"] = target
        self._liste_thresholds = [0] * (len (self._attributs.keys ()) - 1)


    def ranking_function_constructor(self,motclef):
        pass


    def ranked_attributs(self,motclef,nb=1):
        if not motclef in self._liste_mesures:
            return None
        ranker = self.ranking_function_constructor(motclef)
        scores = []
        L = range(0,len(self._attributs.keys())-2)
        if(self._subsets == None or nb == 1):
            C = combinations(L,nb)
        else:
            print("nb = ",nb)
            print(self._subsets)
            C = self._subsets[nb]
        for i in C:
            K = []
            for j in i:
                K.append(j)
            if(self._liste_thresholds[self._liste_mesures.index(motclef)] == 0 or ranker(tuple(K))>=self._liste_thresholds[self._liste_mesures.index(motclef)]):
                t = i,ranker(tuple(K))
            else:
                t = i,(-1)
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


