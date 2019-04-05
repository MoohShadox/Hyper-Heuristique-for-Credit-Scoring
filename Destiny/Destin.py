from Destiny.DataSets.german_dataset import load_german_dataset
from Destiny.Embedded_Thresholding import Embedded_Thresholding
from Destiny.RankingFunctions.Final.Distances_Measures import Distances_Measures

from Destiny.RankingFunctions.Final.Information_Measure import Information_Measure
from Destiny.RankingFunctions.Final.MesureDeConsistance import MesureDeConsistance
from Destiny.RankingFunctions.Final.MesureDeDependance import MesureDeDependance
from Destiny.RankingFunctions.Final.PrecisionClassification import PrecisionClassification
from Destiny.Tresholding import Tresholding
import numpy as np


class Destiny:
    #C'est simple pour demander un ranking donné tu précise la lettre suivi de l'indice donc D0 pour Chi, I1 pour le gain d'information etc...
    #Sinon on peut indexer en utilisant H et un chiffre qui commence a 0
    mesures_distance = ["FScore","ReliefF","FCS"]
    mesures_information = [ 'GainInformation' , "GainRatio" , "SymetricalIncertitude" , "MutualInformation" , "UH" , "US" , "DML"]
    mesures_classification = ["RF",  "AdaBoost"]
    mesures_consistance = ['FCC']
    mesures_dependance = ["RST"]
    #mesures_classification = ["BN","RF","LSVM","RBFSVM","GaussianProcess","AdaBoost","QDA","KNN","DTC","MLP"]

    def __init__(self):
        self.__data,self.__target = None,None
        self.__mesures = {}
        self.__Threshold = 0
        self.__nom_mesures = {}
        self.subsetgenerated = None
        self.__mesures_anterieure = {}
        self.__matrices_redondaces, self.__matrices_importances = {} , {}
        self.__mesures["D"],self.__nom_mesures["D"] = Distances_Measures(),Destiny.mesures_distance
        self.__mesures["I"],self.__nom_mesures["I"] = Information_Measure(),Destiny.mesures_information
        self.__mesures["C"],self.__nom_mesures["C"] = PrecisionClassification(),Destiny.mesures_classification
        self.__mesures["Co"],self.__nom_mesures["Co"] = MesureDeConsistance(),Destiny.mesures_consistance
        self.__mesures["De"],self.__nom_mesures["De"] = MesureDeDependance(),Destiny.mesures_dependance



    def __copy__(self):
        D = Destiny()
        D.__mesures["D"] = self.__mesures["D"]
        D.__mesures["I"] = self.__mesures["I"]
        D.__mesures["C"] = self.__mesures["C"]
        D.__mesures["Co"] = self.__mesures["Co"]
        D.__mesures["De"] = self.__mesures["De"]

    #L est une liste de types de mesures par exemple ['C' ,'Ce', 'De']
    def GestionSubsets(self,L,Borne = None):
        for i in L:
            self.__mesures[i].CreateSubsets(Borne)

    def getDataset(self):
        return self.__data,self.__target

    def Projection(self,subset):
        #Ancienne projection :
        #D = self.__mesures["D"].ranking_function_constructor("FCS")(subset)
        #I = self.__mesures["I"].ranking_function_constructor("US")(subset)

        #Nouvelle Projection :
        D = self.MinimumRMaxS (subset,"Distance")
        I = self.MinimumRMaxS (subset , "Information")
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


    def MinimumRMaxS(self,subset,mc):
        masque = self.__data.shape[1]*[0]
        for i in subset:
            masque[i] = 1
        masque = np.array(masque)
        return masque.transpose().dot(self.__matrices_redondaces[mc]).dot(masque)-masque.dot(self.__matrices_importances[mc])


    def setMatricesImportanceRedondance(self,data,target):
        self.__mesures["I"].fit (data , target)
        data = data.transpose ()
        d = list (data)
        d.append (target)
        d = np.array (d)
        #data = data.transpose ()
        c = np.corrcoef (d)
        c = c.transpose()
        self.__matrices_importances["Distance"] = c[-1]
        self.__matrices_redondaces["Distance"] = c[0:-1]
        self.__matrices_redondaces["Distance"] = self.__matrices_redondaces["Distance"].transpose()[:-1]
        self.__matrices_importances["Distance"] = self.__matrices_importances["Distance"].transpose ()[:-1]
        f = self.__mesures["I"]
        m = np.ones((len(data),len(data)))
        imp = np.ones((len(data)))
        for i in range(0,len(data)):
            C1 = f.getEntropy([i])
            for j in range(0,len(data)):
                C2 = f.getEntropy([j])
                C3 = f.getEntropySachant([j],[i])
                m[i][j] = C1 + C2 - C3
            C2 = f.getEntropy([-1])
            C3 = f.getEntropySachant([i],[-1])
            imp[i] = C1 + C2 - C3
        self.__matrices_redondaces["Information"] = m
        self.__matrices_importances["Information"] = imp
        unit = np.ones(self.__matrices_importances["Distance"].shape[0])
        for i in self.__matrices_importances:
            self.__matrices_importances[i] = unit - self.__matrices_importances[i]/self.__matrices_importances[i].sum()
        return self.__matrices_redondaces,self.__matrices_importances


    def ThresholdMeasures(self,seuil):
        self.__mesures_anterieure = self.__mesures.copy()
        self.__Threshold = seuil
        for i in self.__mesures.keys():
            self.__mesures[i].setThresholdsAutomatiquement(self.__Threshold)
        pass



    def fit(self,X,Y):
        self.__data ,self.__target = X, Y
<<<<<<< HEAD
        m1 , m2 = self.setMatricesImportanceRedondance(X,Y)

=======
        T = Tresholding()
        T.fit(X,Y)
        #self.__Threshold = T.getTreshold(X,Y)
        self.__Threshold = 0.4
        print(self.__Threshold)
>>>>>>> master
        for i in self.__mesures.keys():
            self.__mesures[i].fit (X , Y)
            print(i," fini")
            if(self.subsetgenerated == None):
                self.__mesures[i].setMatrix(m1,m2)
                self.subsetgenerated = self.__mesures[i].CreateSubsets(borne = 1000)
            else:
                self.__mesures[i].setSubsets(self.subsetgenerated)
        T = Tresholding ()
        T.fit (X , Y)
        self.ThresholdMeasures(T.getTreshold(X,Y))

    def getMatriceImportanceRedondance(self):
        return self.__matrices_redondaces,self.__matrices_importances

    def test(self):
        D = {}
        for i in self.__mesures.keys():
            print("Utilisation de ", i)
            self.__mesures[i].rank_with(n=1)
            self.__mesures[i].rank_with(n=2)
            print(self.__mesures[i].rank_with(n=3))




data,target = load_german_dataset()
Dest = Destiny()
Dest.fit(data,target)
D = Embedded_Thresholding()
D.fit(data,target)
Dest.test()
