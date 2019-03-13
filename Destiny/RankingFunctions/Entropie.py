
import math
import numpy as np
import time

class Entropy:

    mesures_existantes = ["Entropie" , 'GainInformation' , "GainRatio" , "SymetricalIncertitude" ,
                          "MutualInformation" , "UH" , "US" , "DML"]
    @staticmethod
    def h(*x):
        liste_vecteurs = []
        for i in x:
            liste_vecteurs.append (i)
        dict_valeurs = {}
        for i in range (0 , len (liste_vecteurs[0])):
            t = []
            for v in liste_vecteurs:
                t.append (v[i])
            tu = tuple (t)
            dict_valeurs[tu] = dict_valeurs.get (tu , 0) + 1
        nb_samples = len (liste_vecteurs[0])
        entropie = 0
        for i in dict_valeurs.keys ():
            dict_valeurs[i] = dict_valeurs[i] / nb_samples
            entropie = entropie + dict_valeurs[i] * math.log (dict_valeurs[i] , 2)
        return -entropie

    def __init__(self):
        self.__attributs = None
        self.__nb_samples = 0
        self.__calculated_measures = {}
        self.__entropy_calculated = []

    def fit(self,X,Y):
        self.__attributs = {}
        self.__nb_samples = len(X)
        A = X.transpose()
        cpt = 0
        for i in A:
            self.__attributs[cpt] = i
            cpt = cpt + 1
        self.__attributs["C"] = Y

    def get_prob(self,param):
        n = 0
        for i in range(0,self.__nb_samples):
            b = True
            for c in param.keys():
                k = c
                if (c != "C" and type(c)!=int):
                    k = int (c.replace ("A" , ""))
                if (str(self.__attributs[k][i]) != str(param[c])):
                    b = False
                    break
            if(b):
                n = n + 1
        return n/self.__nb_samples

    def lister_valeur(self,numero):
        list_val = []
        if (numero != "C"):
            try:
                numero = int (numero.replace ("A" , ""))
            except(Exception):
                pass
        for i in self.__attributs[numero]:
            if not i in list_val:
                list_val.append (i)
        return list_val

    def get_entropie_att(self,numero):
        list_val = self.lister_valeur(numero)
        somme= 0
        if (numero != "C"):
            numero = int (numero.replace ("A" , ""))
        for i in list_val:
            Dict = {}
            Dict[numero] = i
            somme = somme + self.get_prob(Dict)*math.log(self.get_prob(Dict),2)
        return -somme

    # Ecris a partir de wikipedia (VALIDE)
    def get_entropie_att_sachant2(self , numero2 , numero1):
        if(numero1 != "C"):
            numero1 = int(numero1.replace("A",""))
        if (numero2 != "C"):
            numero2 = int (numero2.replace ("A" , ""))
        HXY = Entropy.h(self.__attributs[numero1],self.__attributs[numero2])
        HY = Entropy.h(self.__attributs[numero1])
        return HXY - HY

    def gain_information(self,numero):
        if(numero=="C"):
            print('error')
            return
        g = self.get_entropie_att("C")
        lval = self.lister_valeur(numero)
        D = {}
        for val in lval:
            D[val] = []
        for i in range(0,self.__nb_samples):
            D[self.__attributs[int(numero.replace("A",""))][i]].append(self.__attributs["C"][i])
        somme = 0
        for x in D.keys ():
            somme = somme + (len (D[x]) / self.__nb_samples) * Entropy.h (D[x])
        return g-somme

    def gain_ration(self,numero):
        if(numero=="C"):
            return
        h = Entropy.h(self.__attributs[int(numero.replace("A",""))])
        GI = self.gain_information(numero)
        return GI/h

    def incertitude_symetrique(self,numero):
        IG = self.gain_information(numero)
        HC = Entropy.h(self.__attributs["C"])
        HX = Entropy.h(self.__attributs[int(numero.replace("A",""))])
        incerti_sym = 2*(IG/(HC+HX))
        return incerti_sym

    def entropie_sachant2(self,numero1,numero2):
        if (numero1 != "C"):
            numero1 = int (numero1.replace ("A" , ""))
        if (numero2 != "C"):
            numero2 = int (numero2.replace ("A" , ""))
        lva2 = self.lister_valeur (numero2)
        D = {}
        for val in lva2:
            D[val] = []
        for i in range (0 , self.__nb_samples):
            D[self.__attributs[numero2][i]].append (self.__attributs[numero1][i])
        somme = 0
        for x in D.keys ():
            somme = somme + (len (D[x]) / self.__nb_samples) * Entropy.h (D[x])
        return somme

    def mutual_information(self,numero):
        HC = Entropy.h (self.__attributs["C"])
        HX = Entropy.h (self.__attributs[int (numero.replace ("A" , ""))])
        HCS = Entropy.h(self.__attributs[int (numero.replace ("A" , ""))],self.__attributs["C"])
        return HC+HX-HCS

    def Us_index(self,numero):
        HX = Entropy.h (self.__attributs[int (numero.replace ("A" , ""))])
        return self.mutual_information(numero)/HX

    def Uh_index(self,numero):
        HCS = Entropy.h (self.__attributs[int (numero.replace ("A" , ""))] , self.__attributs["C"])
        return self.mutual_information(numero)/HCS

    def DML_index(self,numero):
        A = self.entropie_sachant2(numero,"C")
        B = self.entropie_sachant2("C",numero)
        return B/A



    def ranking_function_constructor(self,motclef):
        ranker = None
        if(motclef=="Entropie"):
            ranker = self.get_entropie_att
        elif(motclef=="GainInformation"):
            ranker = self.gain_information
        elif(motclef=="GainRatio"):
            ranker = self.gain_ration
        elif(motclef=="SymetricalIncertitude"):
            ranker = self.incertitude_symetrique
        elif(motclef=="MutualInformation"):
            ranker = self.mutual_information
        elif(motclef == "UH"):
            ranker = self.Uh_index
        elif(motclef == "US"):
            ranker = self.Us_index
        elif(motclef == "DML"):
            ranker = self.DML_index
        return ranker

    def ranked_attributs(self,motclef):
        rank = self.ranking_function_constructor(motclef)
        L = []
        for i in range(0,len(self.__attributs)-1):
            t = i,rank("A"+str(i))
            L.append(t)
        L.sort(key=lambda x:x[1],reverse=True)
        return L

    def rank_all(self):
        for mc in Entropy.mesures_existantes:
            self.__calculated_measures[mc] = self.ranked_attributs(mc)
        return self.__calculated_measures








