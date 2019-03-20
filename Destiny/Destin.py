from Destiny.RankingFunctions import Entropie
from Destiny.RankingFunctions.Final.Distances_Measures import Distances_Measures

from Destiny.RankingFunctions.Final.Information_Measure import Information_Measure
from Destiny.RankingFunctions.Final.MesureDeConsistance import MesureDeConsistance
from Destiny.RankingFunctions.Final.MesureDeDependance import MesureDeDependance
from Destiny.RankingFunctions.Final.PrecisionClassification import PrecisionClassification
from Destiny.Tresholding import Tresholding


class Destiny:
    #C'est simple pour demander un ranking donné tu précise la lettre suivi de l'indice donc D0 pour Chi, I1 pour le gain d'information etc...
    #Sinon on peut indexer en utilisant H et un chiffre qui commence a 0
    #mesures_distance = ["Chi","FScore","ReliefF","FCS"]
    mesures_distance = ["FScore" , "ReliefF" , "FCS"]
    mesures_information = ["Entropie" , 'GainInformation' , "GainRatio" , "SymetricalIncertitude" , "MutualInformation" , "UH" , "US" , "DML"]
    mesures_dependance = ["RST"]
    mesures_consistance = ['FCC']
    mesures_classification =  ["BN" , "RF" , "KNN" , "AdaBoost"]

    def __init__(self):
        self.__data,self.__target = None,None
        self.__mesures = {}
        self.__Threshold = 0
        self.__nom_mesures = {}
        self.__precedent_measures = {}
        self.__mesures["D"],self.__nom_mesures["D"] = Distances_Measures(),Destiny.mesures_distance
        self.__mesures["I"],self.__nom_mesures["I"] = Information_Measure(),Destiny.mesures_information
        self.__mesures["C"],self.__nom_mesures["C"] = PrecisionClassification(),Destiny.mesures_classification
        self.__mesures["Co"],self.__nom_mesures["Co"] = MesureDeConsistance(),Destiny.mesures_consistance
        self.__mesures["De"],self.__nom_mesures["De"] = MesureDeDependance(),Destiny.mesures_dependance


    #L est une liste de types de mesures par exemple ['C' ,'Ce', 'De']
    def GestionSubsets(self,L,Borne = None):
        for i in L:
            self.__mesures[i].CreateSubsets(Borne)

    def getDataset(self):
        return self.__data,self.__target

    def Projection(self,subset):
        D = self.__mesures["D"].ranking_function_constructor("FCS")(subset)
        I = self.__mesures["I"].ranking_function_constructor("GainRatio")(subset)
        DE = self.__mesures["De"].dependence(subset)
        Co = self.__mesures["Co"].fcc(subset)
        return (D,I,DE,Co)

    def getMegaHeuristique(self,ids,nb):
        D = {}
        Lmotsclefs = []
        for id in ids:
            cpt = 0
            motclef = None
            tp = None
            for i in self.__nom_mesures.keys():
                cptlocal = 0
                for j in self.__nom_mesures[i]:
                    cpt = cpt + 1
                    cptlocal = cptlocal + 1
                    if(id == str(i)+str(cptlocal) or id == "H"+str(cpt)):
                        motclef = j
                        tp = i
                        break
                if(motclef != None):
                    break
            L = []
            L.append(motclef)
            Lmotsclefs.append(motclef)
            D.update(self.__mesures[tp].rank_with(L,n=nb))
        DDD = {}
        for i in D[nb]:
            if i in Lmotsclefs:
                DDD[i] = D[nb][i]
        return DDD

    def fit(self,X,Y):
        self.__data ,self.__target = X, Y
        T = Tresholding()
        T.fit(X,Y)
        self.__Threshold = T.getTreshold(X,Y)
        print(self.__Threshold)
        for i in self.__mesures.keys():
            self.__mesures[i].fit (X , Y)
            print(i," fini")
            self.__mesures[i].setThresholdsAutomatiquement (self.__Threshold)
        print("Fin du fit")




    def getprecdico(self):
        return self.__precedent_measures

    def test(self):
        D = {}
        for i in self.__mesures.keys():
            print("Utilisation de ", i)
            print(self.__mesures[i].rank_with(n=1))





