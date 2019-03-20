from sklearn.datasets import make_classification

from Destiny.DataSets import german_dataset
from Destiny.DataSets.australian_dataset import load_australian_dataset
from Destiny.DataSets.load_spambase_dataset import load_spambase_dataset
from Destiny.Destin import Destiny
from Destiny.RankingFunctions import ReliefF
from Destiny.RankingFunctions.Final.ChiMesure import chi
from Destiny.RankingFunctions.Final.FScore import FScore
from Nature2.Nature import Nature
from sklearn.datasets import load_wine
import pandas as pd
from Destiny.DataSets import load_kaggle_dataset
#data, target = german_dataset.load_german_dataset()
#data,target = wine.data, wine.target
data,target = load_australian_dataset()
print("dumped")
DM= Destiny()
DM.fit(data,target)
DM.test()
Nature.init(DM)
for i in range(10):
    Nature.evolve()
    print("Le génome alpha",Nature.actualalpha.incarnation)
    print("la precision",Nature.actual_precision)