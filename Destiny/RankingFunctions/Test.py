from itertools import combinations

from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from Destiny.DataSets import german_dataset
from Destiny.DataSets.australian_dataset import load_australian_dataset
from Destiny.DataSets.load_promoters_dataset import load_promoter_dataset
from Destiny.Destin import Destiny
from Destiny.Evaluateur_Precision import Evaluateur_Precision
from Nature2.Nature import Nature
import time

#data,target = wine.data, wine.target
data,target = load_promoter_dataset()
DM= Destiny()
DM.fit(data,target)
print("fin du fit")
#print("le seul threshold généré est : ",DM.generer_un_seul_threshold(0))
#E.train(AdaBoostClassifier())
print("rapport de classification : ")
print(DM.rapport_heuristique(5,AdaBoostClassifier()))
#for i in combinations(L,10):
#
#    t = time.time_ns()
#    print(E.Evaluer(list(i)))
#    if(time.time_ns() - t == 0):
#        print("Combinaison donnant 0 : ",i)

#Nature.init(DM)
#print("debut de l'évolution")
#for i in range(20):
#    a=time.time()
#    Nature.evolve()
#    print("temps:",time.time()-a)
#    print("Le génome alpha",Nature.actualalpha.incarnation)
#    print("la precision",Nature.actual_precision)


#DM= Destiny()
#print("Data : ", data.shape,"Target : ", target.shape)
#DM.fit(data,target)
#Nature.init(DM)
#for i in range(20):
#    Nature.evolve()
#    print("Le génome alpha",Nature.actualalpha.incarnation)
#    print("la precision",Nature.actual_precision)
