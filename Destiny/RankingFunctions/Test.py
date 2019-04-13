from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier

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
for i in range(20):
    a=time.time()
    Nature.evolve()
    print("temps:",time.time()-a)
    print("Le génome alpha",Nature.actualalpha.incarnation)
    print("la precision",Nature.actual_precision)
