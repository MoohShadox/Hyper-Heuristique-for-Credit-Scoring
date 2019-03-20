import pandas as pd
import numpy as np
from Destiny.Evaluateur_Precision import Evaluateur_Precision
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier

train = pd.read_csv('train.csv')
print("dumped")
target = train['target']
features = [c for c in train.columns if c not in ['ID_code', 'target']]
train = train[features]
train = np.array(train)
target = np.array(target)
#print(np.array(train).shape)
#E = Evaluateur_Precision(train,target)
#E.train(RandomForestClassifier(n_estimators=30))
#print(E.score())#



from Destiny.Destin import Destiny
from Nature2.Nature import Nature

DM= Destiny()
DM.fit(train,target)
DM.test()
Nature.init(DM)
for i in range(10):
    Nature.evolve()
    print("Le génome alpha",Nature.actualalpha.incarnation)
    print("la precision",Nature.actual_precision)