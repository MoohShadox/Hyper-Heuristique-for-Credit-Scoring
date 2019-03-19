import time
import timeit

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import *
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score



class Evaluateur_Precision:
    #Prend en paramètre un vecteur d'attribut et initialise les attributs utile a l'évaluation de précision
    def __init__(self,data,target,masque=None):
        self.__target = target
        self.__data = data
        self.__vecteur_attributs = masque
        scaler_model = preprocessing.StandardScaler ().fit (data)
        scaler_model.transform(data)
        self.__data_train, self.__data_test, self.__target_train , self.__target_test \
            = train_test_split(data, target, test_size=0.4,random_state=0)

    def afficher_data(self):
        print(self.__data_train)

    def train(self,modele):
        self.__model = modele.fit(self.__data_train,self.__target_train)

    def vecteur_precision(self):
        return cross_val_score(self.__model,self.__data,self.__target,cv = 10)

    def score(self):
        return self.__model.score(self.__data_test,self.__target_test)

    def Evaluer(self,numeros):
        X,Y,Z,W = self.__data_train, self.__data_test, self.__target_train , self.__target_test
        masque = np.array(len(self.__data_test[0]) * [False])
        for i in numeros:
            masque[i] = True
        self.masquer(masque)
        S = self.score()
        self.__data_train , self.__data_test , self.__target_train , self.__target_test = X,Y,Z,W
        return S

    def masquer(self,masque):
        self.__data_test = self.__data_test.transpose ()
        self.__data_test = self.__data_test[masque]
        self.__data_test = self.__data_test.transpose ()
        self.__data_train = self.__data_train.transpose ()
        self.__data_train = self.__data_train[masque]
        self.__data_train = self.__data_train.transpose ()
        self.train(self.__model)


