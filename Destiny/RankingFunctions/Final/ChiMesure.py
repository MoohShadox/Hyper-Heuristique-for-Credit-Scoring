from sklearn.feature_selection import chi2
import numpy as np

from Destiny.RankingFunctions import Dimension_Reductor


class chi:
    def __init__(self,data,target):
        self.__data = data
        self.__target = target
        self.__scores = {}
        sc = chi2(data,target)
        for i in range(0,len(sc[0])-1):
            t = i,sc[0][i]
            self.__scores[i] = t

    def score(self,x):
        if(len(x)>1):
            DR = Dimension_Reductor.Dimension_Reductor ()
            DR.fit (self.__data , self.__target)
            L = DR.getPCA (x)
            LL = []
            LL.append (L)
            LL = np.array (LL)
            LL = LL.transpose ()
            R = chi2(LL , self.__target)
            return R[0][0]
        else:
            return self.__scores[x[0]][1]


