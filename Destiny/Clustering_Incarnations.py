from itertools import combinations

from Destiny.Destin import Destiny
from sklearn.cluster import k_means

class Clustering_Incarnations:
    def __init__(self):
        self.__population = None
        self.__projections = []
        self.__destiny = Destiny()
        self.clusters = {}
        self.alphas_locaux = []

    def fit(self,X,Y):
        self.__destiny.fit(X,Y)

    def ajouter_population(self,X):
        self.__population = X


    def setDestiny(self,D):
        self.__destiny = D

    def projeter(self):
        print("population en entrée ", self.__population)
        self.__projections = []
        for i in self.__population:
            self.__projections.append(self.__destiny.Projection(i))
        return self.__projections

    @staticmethod
    def carreProjection(projection):
        s = 0
        for i in projection:
            s = s + i*i
        return s

    @staticmethod
    def maxCarreProjection(liste_projections):
        m = 0
        im = None
        for i in liste_projections:
            if (Clustering_Incarnations.carreProjection(i) >= m):
                m = Clustering_Incarnations.carreProjection(i)
                im = i
        return im


    def clusteriser(self):
        KM = k_means(self.__projections,n_clusters=3)
        cpt = 0
        Rez = {}
        for i in KM[1]:
            W = []
            t = tuple(self.__population[cpt])
            W.append(t)
            Rez[i] = Rez.get(i,[]) + W
            cpt = cpt + 1
        self.clusters = Rez
        self.alphas_locaux = (len(self.clusters.keys()))*[0]
        for i in self.clusters:
            C = Clustering_Incarnations.maxCarreProjection(self.clusters[i])
            self.alphas_locaux[i] = C


#from Destiny.DataSets import german_dataset
#data, target = german_dataset.load_german_dataset()
#L = range(0,len(data[0])-2)
#CI  = Clustering_Incarnations()
#CI.fit(data,target)
#K = []
#print("Test pour ",len(list(combinations(L,2)))," elements ")
#for i in combinations(L,2):
#    K.append(list(i))
#print(K)
#CI.ajouter_population(K)
#
#for i in CI.projeter():
#    print(i)
#
#CI.clusteriser()
