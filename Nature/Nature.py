import re
import random
import itertools
import Genome
class Nature:
    Pstop=0.5
    maxA = 4
    maxH = 5
    maxP = 1000
    maxS = 10
    strat = []
    population = []
    actualalpha = ""

    @classmethod
    def csm(cls,G0I,G0A,Strat):
        st=""
        exp="\d[h\d]+/"
        gi=re.findall(exp,G0I)
        ga=re.findall(exp,G0A)
        for s,s2 in itertools.zip_longest(gi,ga):
            if(s!=None):
                p = random.random()
                if(p<Strat[0]):
                    st=st+"S"
                else:
                    if(p<Strat[1]):
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
        st = st + "h" + str(k)
        while(random.random()<cls.Pstop):
            k = random.randint(1, 10)
            st = st + "h" + str(k)
        return st

    @classmethod
    def MergeH(cls,gi, ga):
        return gi + ga[1:]

    @classmethod
    def PseudoTransoducteur(cls, GOI, GOA, CSM):
        st = ""
        exp = "(\d[h\d]+)/"
        exp2 = "S|CI|CO|MI|MO"
        gi = re.findall(exp, GOI)
        ga = re.findall(exp, GOA)
        csm = re.findall(exp2, CSM)
        modif = itertools.zip_longest(gi, ga, csm)

        for i, a, c in modif:
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

        VG = Genome.Genome()

        if (len(G.identity) == 0):

            ppp = random.randint(0, 1)
            if (ppp == 1):
                VG.identity = cls.PseudoTransoducteur("1h1/1h1/1h1/1h1/1h1/1h1", "", "MOMOMOMOMOMOMOMO")
            else:
                VG.identity = cls.PseudoTransoducteur("1h1/1h1/1h1/1h1/1h1", "", "MOMOMOMOMOMOMO")
        else:
            VG=G

        return VG

    @classmethod
    def monoevolv(cls, VGOI, VGOA, strat):
        st = cls.csm(VGOI.identity, VGOA.identity, strat)
        GN = Genome.Genome()
        GN.identity = cls.PseudoTransoducteur(VGOI.identity, VGOA.identity, st)
        VGN = cls.validate(GN)
        #print(VGN.identity)
        return VGN

    @classmethod
    def eludeAlpha(cls):
        #provisoire pour le test:
        return cls.population[random.randint(0,cls.maxP-1)]

    @classmethod
    def init(cls):

        cls.strat=[[0.2,0.6,0.5,0.8],[0.2,0.6,0.5,0.8],[0.2,0.6,0.5,0.8],[0.2,0.6,0.5,0.8],[0.2,0.6,0.5,0.8],[0.2,0.6,0.5,0.8],
               [0.2, 0.6, 0.5, 0.8],[0.2,0.6,0.5,0.8],[0.2,0.6,0.5,0.8],[0.2,0.6,0.5,0.8]]
        cls.population=[]

        VNG=Genome.Genome()
        for i in range(cls.maxP):
            cls.population.append(cls.monoevolv(VNG,VNG,cls.strat[random.randint(0,cls.maxS-1)]))

        cls.actualalpha=cls.eludeAlpha()

    @classmethod
    def evolve(cls):
        for i in range(cls.maxP):
            cls.population[i] = cls.monoevolv(cls.population[i], cls.actualalpha,
                                                cls.strat[random.randint(0, cls.maxS - 1)])
            #print(cls.population[i])
        cls.actualalpha = cls.eludeAlpha()

