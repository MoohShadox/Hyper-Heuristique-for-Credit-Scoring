import random
from datetime import time
from itertools import *

import numpy as np
import math

from Destiny.DataSets.german_dataset import load_german_dataset
from Destiny.DataSets.load_promoters_dataset import load_promoter_dataset
from Destiny.DataSets.musk_dataset import load_musk_dataset


class Tresholding:

    def __init__(self):
        self.__data = None
        self.__target = None
        self.__min_max = {}
        self.__valeurs_classe = {}
        self.__treshold_percentage = {}
        self.__proportions_classes = {}
        self.__discriminants_ficher = None
        self.__volume_overlap_region = None
        self.__pourcentage_dehors_overlap = None
        self.__nb_features = 0
        self.__alpha = 0.80

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
#
    def getAttributClasse(self,classe,attribut):
        L= []
        for j in self.__valeurs_classe[classe]:
            L.append(j[attribut])
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


    def F1(self,attributs):
        nb_features = len(self.__data[0])
        if(self.__discriminants_ficher == None):
            self.__discriminants_ficher = {}
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
        for i in attributs:
            if(self.__discriminants_ficher[i]>m):
                m = self.__discriminants_ficher[i]
        return m

    def MinF(self,feature,classe):
        K = []
        K.append(feature)
        K.append(classe)
        if not (tuple(K) in self.__min_max.keys() ) :
            M = self.getAttributClasse(classe,feature)
            M = np.array(M)
            L = []
            L.append(M.min())
            L.append(M.max())
            self.__min_max[tuple(K)] = L
        return self.__min_max[tuple(K)][0]

    def MaxF(self,feature,classe):
        K = []
        K.append (feature)
        K.append (classe)
        if not (tuple (K) in self.__min_max.keys ()):
            M = self.getAttributClasse (classe , feature)
            M = np.array (M)
            L = []
            L.append (M.min ())
            L.append (M.max ())
            self.__min_max[tuple (K)] = L
        return self.__min_max[tuple (K)][1]


    def MINMAX(self,feature,classe1,classe2):
        return min(self.MaxF(feature,classe1),self.MaxF(feature,classe2))

    def MAXMIN(self,feature,classe1,classe2):
        return max(self.MinF(feature,classe1),self.MinF(feature,classe2))

    def MINMIN(self,feature,classe1,classe2):
        return min(self.MinF(feature,classe1),self.MinF(feature,classe2))

    def MAXMAX(self,feature,classe1,classe2):
        return max(self.MaxF(feature,classe1),self.MaxF(feature,classe2))

    def F2(self,attributs):
        if(self.__volume_overlap_region == None):
            self.__volume_overlap_region = {}
            L = list(self.__valeurs_classe.keys())
            produit = 1
            for j in range (0 , len (self.__data[0])):
                C = combinations(L,2)
                for i in C:
                    A = self.MINMAX(j,i[0],i[1]) - self.MAXMIN(j,i[0],i[1])
                    A = max(0,A)
                    B = self.MAXMAX(j,i[0],i[1]) - self.MINMIN(j,i[0],i[1])
                    produit = produit * A/B
                    self.__volume_overlap_region[j] = produit
        p = 1
        for i in attributs:
            p = p * self.__volume_overlap_region[i]
        return p


    def F3(self,attributs):
        if(self.__pourcentage_dehors_overlap == None):
            self.__pourcentage_dehors_overlap = {}
            L = list (self.__valeurs_classe.keys ())
            s = len(self.__data)
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
                self.__pourcentage_dehors_overlap[j] = cpt / s
        m = 0
        for i in attributs:
            if(self.__pourcentage_dehors_overlap[i]>m):
                m = self.__pourcentage_dehors_overlap[i]
            if(m==1):
                break
        return m



    def setThresholdinDestiny(self,D,data,target):
        pass

    def Energie(self,L):
        e = self.__alpha * ((1 / self.F1 (L)) + self.F2 (L) + (1 / self.F3 (L))) / 3 + (1 - self.__alpha) * (len (L) / self.__nb_features)
        #e = self.__alpha * (1/self.F2(L)) + (1-self.__alpha)*(len(L) / self.__nb_features)
        return e

    def GenererListeRandom(self):
        r = random.randint (1 , self.__nb_features - 1)
        L = r * [-1]
        for i in range(0,r):
            k = random.randint(0,self.__nb_features-1)
            while(k in L):
                k = random.randint (0 , self.__nb_features-1)
            L[i] = k
        return L

    def Alteration_Diversification(self,L):
        i = random.randint(1,self.__nb_features-1)
        NL = i*[0]
        print("L = ",L)
        for j in range(0,i):
            if(j < len(L)):
                NL[j] = L[j]
            else:
                k = random.randint(0,self.__nb_features-1)
                while(k in NL):
                    k = random.randint(0,self.__nb_features-1)
                NL[j] = k
        print("NL = ", NL)
        return NL


    def Alteration_Insensification(self,L,T):
        ELMIN = 0
        ELMAX = 0
        augmentation_taille = int(math.log(self.__nb_features,2)*T) + 1
        if(augmentation_taille == 0):
            augmentation_taille = 1
        nouvelle_taille_sup = len(L) + augmentation_taille
        nouvelle_taille_inf = len(L) - augmentation_taille
        NLP,NLM = None,None
        if(nouvelle_taille_inf > 0):
            NLM = nouvelle_taille_inf*[0]
            for w in range(0,nouvelle_taille_inf):
                NLM[w] = L[w]
        if(nouvelle_taille_sup <= self.__nb_features):
            NLP = nouvelle_taille_sup*[0]
            i = 0
            while(i<len(L)):
                NLP[i] = L[i]
                i = i + 1
            while(i<nouvelle_taille_sup):
                z = random.randint(0,self.__nb_features-1)
                while(z in NLP):
                    z = random.randint (0 , self.__nb_features - 1)
                NLP[i] = z
                i = i + 1
        if(NLP == None):
            EMAX = -1
        else:
            EMAX = self.Energie(NLP)
        if(NLM == None):
            EMIN = -1
        else:
            EMIN = self.Energie(NLM)
        if(EMIN > EMAX):
            return NLM
        else:
            return NLP


    def getTreshold(self,data,target):
        self.__nb_features = len(data[0])
        self.fit(data,target)
        L = self.GenererListeRandom()
        e = self.Energie(L)
        pdivers = 0.5
        T = 1
        emin,percentagemin = 1000,1000
        nb_iterations_pas = 10
        cptcpt = nb_iterations_pas
        ep = 0
        pas = 0.20
        coef_decrementation_temperature = 0.80
        while(T  > 0.01):
            k = random.randint(1,100)
            #Mecanisme de régulation des diversification / intensification
            if(k<pdivers*100):
                NL = self.Alteration_Diversification(L)
            else:
                NL = self.Alteration_Insensification(L,T)
            print("Le nouvel ensemble a estimer est ",NL)
            #Mise a jour de la température
            if(e == ep):
                pdivers = pdivers + 0.05
                if(pdivers == 1):
                    break
            T = T * coef_decrementation_temperature
            if(cptcpt == 0):
                cptcpt = nb_iterations_pas
                T = T - pas
            print("La température est : ",T)
            #Mise a jour de l'energie
            ep = e
            enew = self.Energie(NL)
            if(enew<=e):
                L = NL
                e = enew
            else:
                try:
                    p = math.exp (-((enew - e) / T))
                except(OverflowError):
                    break
                p = int(p*100)
                rr = random.randint(0,100)
                if(p<rr):
                    L = NL
                    e = enew
            print("Et donc la nouvelle energie s'èleve a ",e)
            #Concerver le maximum
            if(e<emin):
                emin = e
                percentagemin = len(L) / self.__nb_features
        return percentagemin







data,target = load_german_dataset()
print(data)
print(target)
T = Tresholding()
T.fit(data,target)
print(T.getTreshold(data,target))


