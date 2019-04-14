import re
import random
import itertools

from Destiny.Clustering_Incarnations import Clustering_Incarnations
from Destiny import Destin as dest
from Destiny.Evaluateur_Precision import Evaluateur_Precision
from Nature2 import Genome
from Nature2 import Fabriquant as fb
import math
import time
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class Nature:
    #Hyper parametres:
    Psupp=0.6
    Pstop=0.4
    maxA = 2
    maxH=13
    maxP = 100
    maxS = 3
    nb_promo=4
    alpha=0
    Tol = 3
    tol_evolutivite = 0.25
    strat = [[0.1, 0.7, 0.5, 0.8], [0.5, 0.3, 0.5, 0.7], [0.3, 0.6, 0.5, 0.7]]

    #Parametres de stockage
    population = []
    actualalpha = None
    DM=None
    modjahidin=[]
    population_clusterised = {}
    alphas_locaux = []
    alpha_global = []
    modele = SVC()
    actual_precision=0
    qualite=0
    PM=1
    evolutivite_inter=0
    actuel_score=0


    @classmethod
    def csm(cls,G0I,G0A,Strat):
        st=""
        exp="\d[H\d]+/"
        gi=re.findall(exp,G0I)
        ga=re.findall(exp,G0A)
        for s,s2 in itertools.zip_longest(gi,ga):
            if(s!=None):
                p = random.random()
                if(p<Strat[0]):
                    st=st+"S"
                else:
                    if(p<Strat[1]*cls.PM):
                        pp=random.random()
                        if(pp<Strat[2]):
                            st=st+"MI"
                        else:
                            st=st+"MO"
                    else:
                        if(s2 != None):
                            if(s[0]==s2[0]):
                                pp = random.random()
                                if (pp < Strat[3]):
                                    st = st + "CI"
                                else:
                                    st = st + "CO"
                            else:
                                st=st+"CO"
                        else:
                            st = st + "MO"
        return st

    @classmethod
    def Grand(cls, G=None):
        st = ""
        if (G is None):
            n = random.randint(1,cls.maxA)
            st = st + str(n)
        else:
            st = st + G[0]
        k = random.randint(1, cls.maxH)
        st = st + "H" + str(k)
        while(random.random()<cls.Pstop):
            k = random.randint(1, cls.maxH)
            st = st + "H" + str(k)
        return st

    @classmethod
    def MergeH(cls,gi, ga):
        return gi + ga[1:]

    @classmethod
    def PseudoTransoducteur(cls, GOI, GOA, CSM):
        st = ""
        exp = "(\d[H\d]+)/"
        exp2 = "S|CI|CO|MI|MO"

        gi = re.findall(exp, GOI)
        ga = re.findall(exp, GOA)
        csm = re.findall(exp2, CSM)
        modif = itertools.zip_longest(gi, ga, csm)
        boola=True
        for i, a, c in modif:
            if(random.random()<cls.Psupp or boola):
                boola=False
                if (gi != None):
                    if (c == "S"):
                        st = st + i + "/"
                    if (c == "MI"):
                        st = st + cls.Grand(i) + "/"
                    if (c == "MO"):
                        st = st + cls.Grand() + "/"
                    if (c == "CI"):
                        st = st + cls.MergeH(i, a) + "/"
                    if (c == "CO"):
                        st = st + a + "/"
        return st

    @classmethod
    def validate(cls, G):
        if (len(G.identity) == 0):
            ppp = random.randint(0, 1)
            if (ppp == 1):
                G.identity = cls.PseudoTransoducteur("1H1/2H3/1H2/2H5/1H6/2H2", "", "MOMOMOMOMOMOMOMO")
            else:
                G.identity = cls.PseudoTransoducteur("1H2H3/2H3/1H4/2H1H4/2H1", "", "MOMOMOMOMOMOMO")
        fab = fb.Fabriquant(G, cls.DM)
        VG = fab.genome
        return VG

    @classmethod
    def monoevolv(cls, VGOI, VGOA, strat):
        st = cls.csm(VGOI.identity, VGOA.identity, strat)
        GN = Genome.Genome()
        GN.identity = cls.PseudoTransoducteur(VGOI.identity, VGOA.identity, st)
        VGN = cls.validate(GN)
        return VGN

    @classmethod
    def eludeAlpha(cls,evolve):
        ko=time.time()
        P = []
        for i in cls.population:
            P.append(i.resultat)
        CI = Clustering_Incarnations()
        CI.setDestiny(Nature.DM)
        CI.ajouter_population(P)
        CI.projeter()
        CI.clusteriser()
        print("temps de clusturing",time.time()-ko)
        Nature.population_clusterised = CI.clusters
        Nature.alphas_locaux = CI.alphas_locaux
        if(len(cls.modjahidin)==cls.nb_promo):
            cls.modjahidin.remove(cls.modjahidin[0])
            cls.modjahidin.append(cls.alphas_locaux)
        else: cls.modjahidin.append(cls.alphas_locaux)
        print("----Moudjahidin",len(cls.modjahidin))
        ko = time.time()
        E = Evaluateur_Precision(Nature.DM.getDataset()[0],Nature.DM.getDataset()[1])
        E.train(Nature.modele)
        print("temps du train",time.time()-ko)
        max=cls.qualite
        for i in Nature.alphas_locaux:
            print("Evaluation de ",i)
            ko = time.time()
            precision=E.Evaluer(i)
            print("temps d'évaluation de ",i,": ",time.time()-ko)
            DD=dest.Destiny()
            c=DD.reguler_par_complexote(precision,len(i))
            if c > max:
                max = c
                cls.alpha_global = i
                cls.actual_precision=precision
        cls.qualite = max
        lesalpha=cls.alphas_locaux
        cls.alphas_locaux=[]
        for k in lesalpha:
            kk=0
            while(kk<len(cls.population)):
                if(cls.population[kk].resultat == list(k)):
                    cls.alphas_locaux.append(cls.population[kk])
                    kk=len(cls.population)+1
                kk=kk+1
        ll=0
        print("----------ALPHA GLOBAL",cls.alpha_global)
        while(ll<len(cls.population)):
            if (cls.population[ll].resultat == list(cls.alpha_global)):
                cls.actualalpha=cls.population[ll]
                ll=len(cls.population)+1
            ll=ll+1
        cls.alter_strategies(CI,evolve)


    @classmethod
    def alter_strategies(cls,C,evolve):
        cls.evolutivite_inter=(C.actul_score - cls.actuel_score)/(cls.maxP*100)
        cls.actuel_score=C.actul_score
        if(evolve):
            if(cls.evolutivite_inter>4):
                cls.PM=max(cls.PM*(1-cls.tol_evolutivite),0.2)
            else:
                cls.PM=min(1,8,cls.PM*(1+cls.tol_evolutivite))
            print("PM",cls.PM)
            print("Evolutivite",cls.evolutivite_inter)



    @classmethod
    def init(cls,D):
        cls.DM=D
        cls.maxH=D.maxH
        cls.alpha=D.alpha
        cls.population=[]
        VNG=Genome.Genome()
        for i in range(cls.maxP):
            cls.population.append(cls.monoevolv(VNG,VNG,cls.strat[random.randint(0,cls.maxS-1)]))
        cls.eludeAlpha(False)

    @classmethod
    def evolve(cls):
        print("evolution")
        aa=0
        bb=0
        opo=time.time()
        for i in range(cls.maxP):
            lp=time.time()
            cls.population[i]=cls.monoevolv(cls.population[i],cls.alphas_locaux[cls.getcluster(cls.population[i])],cls.strat[random.randint(0, cls.maxS - 1)])
            aa=aa+(time.time()-lp)
            lp=time.time()
            cls.population[i] = cls.monoevolv(cls.population[i], cls.actualalpha, cls.strat[random.randint(0, cls.maxS - 1)])
            bb=bb+(time.time()-lp)
        print("temps de premiere monoevolution :",aa)
        print("temps de deuxieme monoevolution :",bb)
        print("temps evolution: ",time.time()-opo)
        print("Election de l'alpha")
        po=time.time()
        cls.eludeAlpha(True)
        print("temps de l'electiond 'un alpha :",time.time()-po)

    @classmethod
    def getcluster(cls,G):
        cpt=0
        trouve=False
        while(cpt<len(cls.population_clusterised)):
            cpt2=0
            while(cpt2<len(cls.population_clusterised[cpt])):
                if(G.resultat == list(cls.population_clusterised[cpt][cpt2])):
                    return cpt
                    cpt2=len(cls.population_clusterised[cpt])+1
                    trouve=True
                cpt2=cpt2+1
            if(trouve):
                cpt = len(cls.population_clusterised) + 1
            cpt=cpt+1


        return -1





def sigmoid(x):
  return 1 / (1 + math.exp(-x))



