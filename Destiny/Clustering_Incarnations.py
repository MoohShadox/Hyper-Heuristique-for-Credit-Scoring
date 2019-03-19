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
        for i in liste_projections:
            if (Clustering_Incarnations.carreProjection(i) > 0):
                m = Clustering_Incarnations.carreProjection(i)
        return m


    def clusteriser(self):
        KM = k_means(self.__projections,n_clusters=10)
        cpt = 0
        Rez = {}
        for i in KM[1]:
            W = []
            t = tuple(self.__population[cpt])
            W.append(t)
            Rez[i] = Rez.get(i,[]) + W
            cpt = cpt + 1
        for i in Rez:
            print(i , " : " , Rez[i])
        self.clusters = Rez
        self.alphas_locaux = len(self.clusters.keys())*[]
        for i in self.clusters:
            self.alphas_locaux[i] = Clustering_Incarnations.maxCarreProjection(self.clusters[i])


from Destiny.DataSets import german_dataset
data, target = german_dataset.load_german_dataset()
L = range(0,len(data[0])-2)
CI  = Clustering_Incarnations()
CI.fit(data,target)
K = []
print("Test pour ",len(list(combinations(L,2)))," elements ")
for i in combinations(L,2):
    K.append(list(i))
print(K)
CI.ajouter_population(K)

for i in CI.projeter():
    print(i)

CI.clusteriser()
