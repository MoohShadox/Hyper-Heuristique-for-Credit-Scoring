from sklearn.feature_selection import f_classif

from Destiny.RankingFunctions import Dimension_Reductor
import numpy as np


class FScore:
    def __init__(self,data,target):
        self.__data = data
        self.__target = target
        self.__scores = {}
        sc = f_classif(data,target)
        cpt = 0
        for i in sc[0]:
            tup = cpt,i
            self.__scores[cpt] = tup
            cpt = cpt + 1

    def score(self, x):
        if (len(x) > 1):
            DR = Dimension_Reductor.Dimension_Reductor()
            DR.fit (self.__data , self.__target)
            L = DR.getPCA (x)
            LL = []
            LL.append (L)
            LL = np.array (LL)
            LL = LL.transpose ()
            R = FScore (LL,self.__target)
            score = (8,R.score([0]))
        else:
            score = self.__scores[x[0]]
        return score[1]





