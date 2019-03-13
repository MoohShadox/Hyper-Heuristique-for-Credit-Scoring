from Destiny.RankingFunctions import ReliefF
from Destiny.RankingFunctions.Final import ChiMesure , FScore

from Destiny.RankingFunctions.Final.FCS import FCS
from Destiny.RankingFunctions.Final.Mesure import Mesure


class Distances_Measures(Mesure):
    liste_mesures = ["Chi","FScore","ReliefF","FCS"]

    def __init__(self):
        super().__init__()
        self._liste_mesures = Distances_Measures.liste_mesures
        self.__data = None
        self.__target = None
        self.__mesures= {}

    def get_mesures(self):
        return self.__mesures

    def ranking_function_constructor(self,motclef):
        print(motclef)
        ranker = self.__mesures[motclef].score
        return ranker

    def fit(self,X,Y):
        super().fit(X,Y)
        self.__data = X
        self.__target = Y
        R = ReliefF()
        CH = ChiMesure.chi(X , Y)
        F = FScore.FScore(X , Y)
        R.fit(X,Y)
        FC = FCS()
        FC.fit(self.__data, self.__target)
        self.__mesures["ReliefF"] = R
        self.__mesures["FScore"] = F
        self.__mesures["Chi"] = CH
        self.__mesures['FCS'] = FC







