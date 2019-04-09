from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier

from Destiny.DataSets import german_dataset
from Destiny.DataSets.australian_dataset import load_australian_dataset
from Destiny.DataSets.load_promoters_dataset import load_promoter_dataset
from Destiny.Destin import Destiny
from Destiny.Evaluateur_Precision import Evaluateur_Precision
from Nature2.Nature import Nature

#data,target = wine.data, wine.target
data,target = load_promoter_dataset()
DM= Destiny()
DM.fit(data,target)
Nature.init(DM)
for i in range(20):
    Nature.evolve()
    print("Le génome alpha",Nature.actualalpha.incarnation)
    print("la precision",Nature.actual_precision)
