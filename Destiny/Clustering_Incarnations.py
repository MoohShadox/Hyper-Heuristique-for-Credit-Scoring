from itertools import combinations

from Destiny.Destin import Destiny
from sklearn.cluster import k_means

class Clustering_Incarnations:
    def __init__(self):
        self.__population = None
        self.__projections = []
        self.__destiny = Destiny()

    def fit(self,X,Y):
        self.__destiny.fit(X,Y)

    def ajouter_population(self,X):
        self.__population = X


    def projeter(self):
        self.__projections = []
        for i in self.__population:
            self.__projections.append(self.__destiny.Projection(i))
        return self.__projections


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



from Destiny.DataSets import german_dataset
data, target = german_dataset.load_german_dataset()
L = range(0,len(data[0])-2)
CI  = Clustering_Incarnations()
CI.fit(data,target)
K = []
print("Test pour ",len(list(combinations(L,2)))," elements ")
for i in combinations(L,2):
    K.append(list(i))
CI.ajouter_population(K)

for i in CI.projeter():
    print(i)

CI.clusteriser()
