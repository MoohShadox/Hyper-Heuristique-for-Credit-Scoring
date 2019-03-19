from Destiny.DataSets import german_dataset
from Destiny.Destin import Destiny
from Destiny.RankingFunctions import ReliefF
from Destiny.RankingFunctions.Final.ChiMesure import chi
from Destiny.RankingFunctions.Final.FScore import FScore

data, target = german_dataset.load_german_dataset()
DM = Destiny()
DM.fit(data,target)