import random
from datetime import time
from itertools import *

import numpy as np
import math

from Destiny.DataSets.musk_dataset import load_musk_dataset


class Tresholding:

    def __init__(self):
        self.__data = None
        self.__target = None
        self.__valeurs_classe = {}
        self.__treshold_percentage = {}
        self.__proportions_classes = {}
        self.__discriminants_ficher = {}
        self.__volume_overlap_region = {}
        self.__pourcentage_dehors_overlap = {}
        self.__nb_features = 0
        self.__alpha = 0.85

    def fit(self,data,target):
        self.__valeurs_classe.clear()
        self.__proportions_classes.clear()
        self.__nb_features = len(data[0])
        self.__data = data
        self.__target = target
        for i in range(0,len(self.__target)):
            L = []
            L.append(tuple(self.__data[i]))
            self.__valeurs_classe[self.__target[i]] = self.__valeurs_classe.get(self.__target[i],[]) + L
            self.__proportions_classes[self.__target[i]] = self.__proportions_classes.get(self.__target[i],0) + 1
        sum = 0
        for j in self.__proportions_classes:
            sum = sum + self.__proportions_classes[j]
        for j in self.__proportions_classes:
            self.__proportions_classes[j] = self.__proportions_classes[j] / sum

    def getAttributClasse(self,classe,attribut):
        L= []
        for j in self.__valeurs_classe[classe]:
            L.append(self.__valeurs_classe[classe][attribut])
        return L

    def masquer(self,numero):
        masque = np.array (len (self.__data[0]) * [False])
        for i in numero:
            ch = i
            masque[ch] = True
        self.__data = self.__data.transpose ()
        self.__data = self.__data[masque]
        self.__data = self.__data.transpose ()
        for i in range(0,len(self.__target)):
            L = []
            L.append(tuple(self.__data[i]))
            self.__valeurs_classe[self.__target[i]] = self.__valeurs_classe.get(self.__target[i],[]) + L
            self.__proportions_classes[self.__target[i]] = self.__proportions_classes.get(self.__target[i],0) + 1
        sum = 0
        for j in self.__proportions_classes:
            sum = sum + self.__proportions_classes[j]
        for j in self.__proportions_classes:
            self.__proportions_classes[j] = self.__proportions_classes[j] / sum


    def F1(self):
        nb_features = len(self.__data[0])
        for i in range(0,nb_features):
            results = {}
            for j in self.__valeurs_classe:
                results[j] = {}
                L = []
                for k in self.__valeurs_classe[j]:
                    L.append(k[i])
                N = np.array(L)
                results[j]["M"] = N.mean()
                results[j]["V"] = N.var()
            numerator = 0
            for j in results:
                for k in results:
                    if(i!=j):
                        numerator = numerator + self.__proportions_classes[k] * self.__proportions_classes[j] * (results[j]["M"] - results[k]["M"])*(results[j]["M"] - results[k]["M"])
            denominateur = 0
            for j in results:
                denominateur = denominateur + results[j]['V'] * self.__proportions_classes[j]
            self.__discriminants_ficher[i] = numerator / denominateur
        m = 0
        for i in self.__discriminants_ficher:
            if(self.__discriminants_ficher[i]>m):
                m = self.__discriminants_ficher[i]
        return m

    def MinF(self,feature,classe):
        M = self.getAttributClasse(classe,feature)
        M = np.array(M)
        return M.min()

    def MaxF(self,feature,classe):
        M = self.getAttributClasse (classe , feature)
        M = np.array (M)
        return M.max ()


    def MINMAX(self,feature,classe1,classe2):
        return min(self.MaxF(feature,classe1),self.MaxF(feature,classe2))

    def MAXMIN(self,feature,classe1,classe2):
        return max(self.MinF(feature,classe1),self.MinF(feature,classe2))

    def MINMIN(self,feature,classe1,classe2):
        return min(self.MinF(feature,classe1),self.MinF(feature,classe2))

    def MAXMAX(self,feature,classe1,classe2):
        return max(self.MaxF(feature,classe1),self.MaxF(feature,classe2))

    def F2(self):
        L = list(self.__valeurs_classe.keys())
        C = combinations(L,2)
        for i in C:
            produit = 1
            for j in range(0,len(self.__data[0])):
                A = self.MINMAX(j,i[0],i[1]) - self.MAXMIN(j,i[0],i[1])
                A = max(0,A)
                B = self.MAXMAX(j,i[0],i[1]) - self.MINMIN(j,i[0],i[1])
                produit = produit * A/B
            self.__volume_overlap_region[i] = produit
        p = 1
        for i in self.__volume_overlap_region.items():
            p = p * i[1]
        return p


    def F3(self):
        L = list (self.__valeurs_classe.keys ())
        s = len(self.__data)
        m = 0
        for j in range (0 , len (self.__data[0])):
            cpt = 0
            d = self.__data.transpose()[j]
            C = combinations (L , 2)
            for i in C:
                MMA = self.MINMAX(j,i[0],i[1])
                MMI = self.MAXMIN(j,i[0],i[1])
                for k in d:
                    if(k<=MMA and k>=MMI):
                        cpt = cpt + 1
            if(cpt/s>m):
                m = cpt/s
            if(m==1):
                break
        return m



    def setThresholdinDestiny(self,D,data,target):
        pass

    def getTreshold(self,data,target):
        self.__nb_features = len(data[0])
        self.fit(data,target)
        L = list(range(0 , self.__nb_features))
        t = 0
        em = 1000
        ez = 1100
        for i in range(1,self.__nb_features , int(math.log(self.__nb_features,2))):
            C = list(combinations(L,i))
            for j in C:
                print(j)
                self.masquer(j)
                e = self.__alpha * ((1/self.F1())) + (1-self.__alpha) * (i/self.__nb_features)
                if(ez==e):
                    self.fit (data , target)
                    break
                ez = e
                if(e<em):
                    em = e
                    t = i/self.__nb_features
                print(em)
                self.fit (data , target)
        return t





train,target = load_musk_dataset()

T = Tresholding()
T.fit(train,target)
print(T.getTreshold(train,target))

