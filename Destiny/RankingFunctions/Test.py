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
#for i in range(DM.maxH):
   # print(DM.getMegaHeuristique(["H"+str(i)],1))
Nature.init(DM)
print("debut de l'évolution")
print("Le génome alpha", Nature.actualalpha.identity)
print("la precision", Nature.actual_precision)
print("qualite", Nature.qualite)
print("taille",Nature.taille)
for i in range(20):
    a=time.time()
    Nature.evolve()
    print("temps:",time.time()-a)
    print("Le génome alpha",Nature.actualalpha.incarnation)
    print("la precision",Nature.actual_precision)
    print("taille", Nature.taille)
    print("qualite",Nature.qualite)

#DM= Destiny()
#print("Data : ", data.shape,"Target : ", target.shape)
#DM.fit(data,target)
#Nature.init(DM)
#for i in range(20):
#    Nature.evolve()
#    print("Le génome alpha",Nature.actualalpha.incarnation)
#    print("la precision",Nature.actual_precision)
#for i in range(DM.maxH):
#    t=DM.generer_un_seul_threshold(i)
#    DM.criteron_heursitique_unique(i,t#)
