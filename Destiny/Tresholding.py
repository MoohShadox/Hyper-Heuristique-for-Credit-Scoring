from itertools import *

import numpy as np
import math
class Tresholding:

    def __init__(self):
        self.__data = None
        self.__target = None
        self.__valeurs_classe = {}
        self.__proportions_classes = {}
        self.__discriminants_ficher = {}
        self.__volume_overlap_region = {}
        self.__alpha = 0.75

    def fit(self,data,target):
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
                print("Les résultats donnés sont ",results)
            numerator = 0
            for j in results:
                for k in results:
                    if(i!=j):
                        numerator = numerator + self.__proportions_classes[k] * self.__proportions_classes[j] * (results[j]["M"] - results[k]["M"])*(results[j]["M"] - results[k]["M"])
            denominateur = 0
            for j in results:
                denominateur = denominateur + results[j]['V'] * self.__proportions_classes[j]
            self.__discriminants_ficher[i] = numerator / denominateur
        print("Au final les discriminants sotn ")
        for i in self.__discriminants_ficher:
            print(i," : ",self.__discriminants_ficher[i])

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
            p = p * i
        return p



from Destiny.DataSets import german_dataset
data, target = german_dataset.load_german_dataset()
FF = Tresholding()
FF.fit(data,target)

print(0.75 * FF.F2()[1] + 0.25 )