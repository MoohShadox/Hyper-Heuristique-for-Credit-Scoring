from Destiny.RankingFunctions import Entropie
from itertools import combinations
import math



class FCS:
    seuil_max= 4
    def __init__(self):
        self.__data = None
        self.__target = None
        self.__classe_feature_r = {}
        self.__feature_feature = {}
        self.__mesures = {}

    def r(self,x,y):
        D = self.__data.transpose()
        if(x < len(D) and y!=-1):
            A = D[x]
        else:
            A = self.__target.transpose()
        if(y < len(D) and y!=-1):
            B = D[y]
        else:
            B = self.__target.transpose()
        HA = Entropie.Entropy.h(A)
        HB = Entropie.Entropy.h(B)
        HAB = Entropie.Entropy.h(A,B)
        rez = (HA + HB - HAB) / HA + HB
        return rez

    def fit(self,data,target):
        self.__data = data
        self.__target = target


    def getFeatureClassCorreclation(self,f):
        if not (f in self.__classe_feature_r.keys()):
            self.__classe_feature_r[f] = self.r(f,-1)
        return self.__classe_feature_r[f]

    def getFeatureFeatureCorreclation(self,f1,f2):
        if not ((f1,f2) in self.__feature_feature.keys()):
            self.__feature_feature[(f1,f2)] = self.r(f1,f2)
        return self.__feature_feature[(f1,f2)]

    def rankingOneByOne(self):
        for i in range(0,len(self.__data[0])):
            self.Score(i)


    def rankingBy(self,n):
        if(n>FCS.seuil_max):
            return None
        if not (n in self.__mesures):
            L = list(range(0,len(self.__data[0])-1))
            for i in combinations(L,n):
                self.Score(i)
            self.__mesures[n] = sorted(self.__mesures[n].items(),key=lambda x:x[1],reverse=True)
        return self.__mesures[n]

    def score(self , x):
        if(len(x)<2):
            return -1
        if not (len(x) in self.__mesures.keys()):
            self.__mesures[len(x)] = {}
        if not (tuple(x) in self.__mesures[len(x)].keys()):
            k = len(x)
            x = sorted(x)
            SFF = 0
            SFC = 0
            cptFF = 0
            cptFC = 0
            for i in x:
                SFC = SFC + self.getFeatureClassCorreclation(i)
                cptFC = cptFC + 1
                for j in x:
                    if(j>i):
                        SFF = SFF  + self.getFeatureFeatureCorreclation(i,j)
                        cptFF = cptFF + 1
            SFF = SFF/cptFF
            SFC = SFC/cptFC
            denom = math.sqrt(k+k*(k-1)*SFF)
            numer = k*SFC
            self.__mesures[len(x)][tuple(x)] = numer/denom
        return self.__mesures[len(x)][tuple(x)]


