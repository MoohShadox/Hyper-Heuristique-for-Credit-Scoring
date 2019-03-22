import random
from itertools import combinations

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import numpy as np

class Embedded_Thresholding:


    borne_complexite = 3000

    def __init__(self):
        self.__modele = AdaBoostClassifier()
        self.__data = None
        self.__data_masque = None
        self._rfecv = None
        self.__target = None
        self.__threshold = 0
        self.__subset_selectionned = {}
        self.__nbfeatures = 0

    def fit(self,X,Y):
        self.__data = X
        self.__target = Y
        self.__data_masque = X
        #rfecv = RFECV (estimator=self.__modele , step=1 , cv=StratifiedKFold (2) ,
        #               scoring='accuracy')
        #rfecv.fit(self.__data,self.__target)
        self.__nbfeatures = len(self.__data.transpose())
        self.__threshold = 100


    def getThresholdEmbedded(self,modele):
        rfecv = RFECV (estimator=modele , step=1 , cv=StratifiedKFold (2) ,
                       scoring='accuracy')
        rfecv.fit(self.__data,self.__target)
        return rfecv.n_features_


    def compute_subset(self,liste_chiffre):
        if not (tuple(liste_chiffre) in self.__subset_selectionned.keys()):
            self.__data_masque = self.__data.transpose ()
            masque = np.array (len (self.__data_masque) * [False])
            for i in liste_chiffre:
                masque[i] = True
            self.__data_masque = self.__data_masque[masque]
            self.__data_masque = self.__data_masque.transpose ()
            rfecv = RFECV (estimator=self.__modele , step=1 , cv=StratifiedKFold (2) ,
                           scoring='accuracy')
            rfecv.fit (self.__data_masque , self.__target)
            self.__subset_selectionned[tuple(liste_chiffre)] = rfecv.ranking_,rfecv.grid_scores_
        return self.__subset_selectionned[tuple(liste_chiffre)]

    def diversifier(self,e):
        r = self.compute_subset(e)
        cpt = 0
        while(tuple(e) in self.__subset_selectionned.keys()):
            r = random.randint(0,len(e)-1)
            j = random.randint(0,self.__nbfeatures-1)
            while(j in e):
                j = random.randint (0 , self.__nbfeatures - 1)
            e[r] = j
            cpt = cpt + 1
            if(cpt == 1000):
                break
        return e

    def intensifier(self,e):
        r = self.compute_subset(e)
        D = {}
        for i in range(0,len(r[0])):
            L = []
            L.append((i,r[1][i]))
            D[r[0][i]] = D.get(r[0][i],[]) + L
        rez = 0
        for i in sorted(list(D.keys())):
            for j in sorted(D[i],key=lambda x:x[1]):
                rez = j[0]
                break
        j = random.randint (0 , self.__nbfeatures - 1)
        while (j in e):
            j = random.randint (0 , self.__nbfeatures - 1)
        e[rez] = j
        return e


    def generer_subset(self,taille,borne = None):
        vinit = np.random.randint(0,self.__nbfeatures,taille)
        p = 0.25
        if(borne == None):
            b = Embedded_Thresholding.borne_complexite
        else:
            b = borne
        for i in range(0,b):
            r = random.randint(0,10)
            if(r/10>p):
                vinit = self.intensifier(vinit)
                if(tuple(vinit) in self.__subset_selectionned):
                    p = p + 0.05
                    vinit = self.diversifier(vinit)
            else:
                vinit = self.diversifier(vinit)
            if(p==1):
                break
        C = list(self.__subset_selectionned.keys())
        self.__subset_selectionned.clear()
        return C
