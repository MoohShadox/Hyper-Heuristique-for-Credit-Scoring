from itertools import combinations

from Destiny.Evaluateur_Precision import Evaluateur_Precision
from Nature2 import Nature as nat
from Destiny.Destin import Destiny
from sklearn.cluster import k_means

class Clustering_Incarnations:
    def __init__(self):
        self.__population = None
        self.__projections = []
        self.__destiny = Destiny()
        self.clusters = {}
        self.alphas_locaux = []
        self.actul_score=0

    def fit(self,X,Y):
        self.__destiny.fit(X,Y)

    def ajouter_population(self,X):
        self.__population = X


    def setDestiny(self,D):
        self.__destiny = D

    def projeter(self):
        self.__projections = []
        for i in self.__population:
            self.__projections.append(self.__destiny.Projection(i))
        return self.__projections

    @staticmethod
    def carreProjection(projection):
        s = 0
        for i in projection:
            s = s + i*i
        D=Destiny()
        return D.reguler_par_complexote(s,len(projection))


    def maxCarreProjection(self,liste_projections):
        m = 0
        im = None
        for i in liste_projections:
            self.actul_score = self.actul_score + Clustering_Incarnations.carreProjection(i)
            if(i not in nat.Nature.alphas_locaux):
                if (Clustering_Incarnations.carreProjection(i) >= m):
                    m = Clustering_Incarnations.carreProjection(i)
                    im = i
        return im

    def maxprecision(self,liste_projection):
        max=0
        im=None
        E = Evaluateur_Precision(nat.Nature.DM.getDataset()[0], nat.Nature.DM.getDataset()[1])
        E.train(nat.Nature.modele)
        for i in liste_projection:
            self.actul_score=self.actul_score+E.Evaluer(i)
            if (i not in nat.Nature.alphas_locaux):
                if (E.Evaluer(i) >= max):
                    max = Clustering_Incarnations.carreProjection(i)
                    im = i
        return im


    def clusteriser(self,train):
        KM = k_means(self.__projections,n_clusters=nat.Nature.nb_cluster)
        cpt = 0
        Rez = {}
        for i in KM[1]:
            W = []
            t = tuple(self.__population[cpt])
            W.append(t)
            Rez[i] = Rez.get(i,[]) + W
            cpt = cpt + 1
        #for i in Rez:
            #print(i , " : " , Rez[i])
        self.clusters = Rez
        self.alphas_locaux = (len(self.clusters.keys()))*[0]
        self.actul_score=0
        for i in self.clusters:
            if not train:
                C = self.maxCarreProjection(self.clusters[i])
            else:
                C=self.maxprecision(self.clusters[i])
            self.alphas_locaux[i] = C


