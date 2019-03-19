from itertools import combinations

from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from Destiny import Evaluateur_Precision , Embedded_Thresholding
import numpy as np
from Destiny.RankingFunctions.Final.Mesure import Mesure


class PrecisionClassification (Mesure):
    # liste_modeles = ["BN","RF","LSVM","RBFSVM","GaussianProcess","AdaBoost","QDA","KNN","DTC","MLP"]
    liste_modeles = ["BN" , "RF" , "KNN" , "AdaBoost"]  # Minimum viable

    def __init__(self):
        super ().__init__ ()
        self.__data = None
        self.__target = None
        self.__evaluateurs = {}
        self.__ranks = {}
        self._liste_mesures = PrecisionClassification.liste_modeles


    def setThresholdsAutomatiquement(self,s=None):
        self.rank_with (n=1)
        E = Embedded_Thresholding.Embedded_Thresholding()
        E.fit(self.__data,self.__target)
        L = []
        for i in PrecisionClassification.liste_modeles:
            try:
                L.append(E.getThresholdEmbedded(PrecisionClassification.modele_generator(i)))
                self._liste_thresholds[PrecisionClassification.liste_modeles.index(i)]  =self.__ranks[1][i][E.getThresholdEmbedded(PrecisionClassification.modele_generator(i))-1][1]
            except(RuntimeError):
                pass
        L = np.array(L)
        s = L.mean()/len(self.__data[0])
        for j in self._calculated_measures[1]:
            if (self._liste_thresholds[self._liste_mesures.index (j)] == 0):
                self._liste_thresholds[self._liste_mesures.index (j)] = self._calculated_measures[1][j][int (s * (len (self._attributs.keys ()) - 1))][1]
        self._calculated_measures.clear ()
        self.__ranks.clear()


    def ranking_function_constructor(self , motclef):
        D = {}
        for i in self.__ranks.keys ():
            for j in self.__ranks[i].keys ():
                if j == motclef:
                    D[tuple (i)] = self.__ranks[i][j]
        return D.get

    def fit(self , data , target):
        super ().fit (data , target)
        self.setup_modeles (data , target)

    def calculate(self , numero , Motclef=None):
        if not (len (numero) in self.__ranks.keys ()):
            self.__ranks[len (numero)] = {}
        if (Motclef != None):
            for i in Motclef:
                if (not i in self.__ranks[len (numero)]):
                    self.__ranks[len (numero)][i] = []
        else:
            for i in PrecisionClassification.liste_modeles:
                if (not i in self.__ranks[len (numero)]):
                    self.__ranks[len (numero)][i] = []
        masque = np.array (len (self.__data[0]) * [False])
        for i in numero:
            ch = i
            masque[ch] = True
        self.masquer (masque)
        for j in self.__evaluateurs.keys ():
            if (Motclef == None or j in Motclef):
                if(self.__evaluateurs[j].score () >= self._liste_thresholds[PrecisionClassification.liste_modeles.index(j)]):
                    self.__ranks[len (numero)][j].append ((tuple(numero) , self.__evaluateurs[j].score ()))
                else:
                    self.__ranks[len (numero)][j].append ((tuple (numero) , -1))
                self.__ranks[len (numero)][j].sort (key=lambda x: x[1] , reverse=True)
        self.setup_modeles (self.__data , self.__target)
        return self.__ranks[len (numero)]

    def getRanks(self):
        return self.__ranks

    def ranked_attributs(self , motclef , nb=1):
        if not motclef in self._liste_mesures:
            return -1
        if (not nb in self.__ranks.keys ()):
            self.__ranks[nb] = {}
        if (not motclef in self.__ranks[nb].keys ()):
            scores = []
            L = range (0 , len (self._attributs.keys ()) - 2)
            if (self._subsets == None):
                C = combinations (L , nb)
            else:
                C = self._subsets[nb]
            for i in C:
                K = []
                for j in i:
                    K.append (j)
                scores = self.calculate (K , [motclef])
        return self.__ranks[nb][motclef]

    def rank_attributs_one_to_one(self):
        for i in range (0 , len (self.__data[0])):
            self.calculate ([i])
        return self.__ranks

    def print_scores(self):
        for i in self.__evaluateurs:
            print (i + " : " + str (self.__evaluateurs[i].score ()))

    def print_multiples_scores(self):
        for i in self.__evaluateurs:
            print (i + " : " + str (self.__evaluateurs[i].vecteur_precision ()))

    def setup_modeles(self , data , target):
        self.__data = data
        self.__target = target
        for m in PrecisionClassification.liste_modeles:
            self.__evaluateurs[m] = Evaluateur_Precision.Evaluateur_Precision (data , target)
            self.__evaluateurs[m].train (PrecisionClassification.modele_generator (m))

    @staticmethod
    def modele_generator(motclef):
        if (motclef == "BN"):
            return GaussianNB ()
        elif (motclef == "DTC"):
            return DecisionTreeClassifier ()
        elif (motclef == "LSVM"):
            return SVC (kernel='linear')
        elif (motclef == "RBFSVM"):
            return SVC (kernel='rbf')
        elif (motclef == "GaussianProcess"):
            return GaussianProcessClassifier ()
        elif (motclef == "AdaBoost"):
            return AdaBoostClassifier ()
        elif (motclef == "QDA"):
            return QuadraticDiscriminantAnalysis ()
        elif (motclef == "KNN"):
            return KNeighborsClassifier ()
        elif (motclef == "RF"):
            return RandomForestClassifier (n_estimators=10)
        elif (motclef == 'MLP'):
            return MLPClassifier ()

    def masquer(self , masque):
        for k in self.__evaluateurs.keys ():
            self.__evaluateurs[k].masquer (masque)


