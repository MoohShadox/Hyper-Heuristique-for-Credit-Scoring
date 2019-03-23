
from Destiny.Destin import Destiny

from Nature2.Nature import Nature

from Destiny.DataSets.madelon_dataset import load__train_dataset

data,target = load__train_dataset()
target = target.transpose()
print("dumped")
DM= Destiny()
DM.fit(data,target)
DM.test()
Nature.init(DM)
for i in range(10):
    Nature.evolve()
    print("Le génome alpha",Nature.actualalpha.incarnation)
    print("la precision",Nature.actual_precision)