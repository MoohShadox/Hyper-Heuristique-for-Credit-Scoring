import time
import xml.dom
from xml.dom import minidom
from xml.dom.minidom import DOMImplementation

from sklearn.svm import SVC

from Destiny.DataSets.load_spambase_dataset import load_spambase_dataset
from Destiny.DataSets.madelon_dataset import load__train_dataset

from Destiny.DataSets.australian_dataset import load_australian_dataset
from Destiny.DataSets.german_dataset import load_german_dataset
from Destiny.DataSets.load_promoters_dataset import load_promoter_dataset
from Destiny.DataSets.musk_dataset import load_musk_dataset
from Destiny.Destin import Destiny
from Nature2.Nature import Nature


def Ecrire_Test_Heuristiques(DM,nd,nroot,modele=SVC()):
    for i in range (0 , Destiny.nb_heuristiques):
        D = DM.rapport_heuristique (i,modele)
        feature_selection = nd.createElement ('FeatureSelection')
        att1 = nd.createAttribute ("Seuil")
        att1.nodeValue = str (D[2])
        att2 = nd.createAttribute ("Subset")
        att2.nodeValue = str (D[1])
        feature_selection.setAttributeNode (att1)
        feature_selection.setAttributeNode (att2)
        att3 = nd.createAttribute("Heuristique")
        att3.nodeValue = str(DM.liste_mesures[i])
        feature_selection.setAttributeNode(att3)
        att4 = nd.createAttribute("NeatPrecision")
        att4.nodeValue = str(D[3])
        feature_selection.setAttributeNode(att4)
        for j in D[0]["weighted avg"]:
            elem = nd.createElement (j)
            val = nd.createTextNode (str (D[0]["weighted avg"][j]))
            elem.appendChild (val)
            feature_selection.appendChild (elem)
        nroot.appendChild (feature_selection)


def Ecrire_Dictionnaire_Heuristiques(DM,nd,nroot):
    Entete = nd.createElement('DictionnaireHeuristiques')
    cpt = 0
    for i in DM.liste_mesures:
        E = nd.createElement("Heuristique")
        att1 = nd.createAttribute("Index")
        att1.nodeValue = str(cpt)
        cpt = cpt + 1
        E.setAttributeNode(att1)
        text = nd.createTextNode(i)
        E.appendChild(text)
        Entete.appendChild(E)
    nroot.appendChild(Entete)

def Ecrire_Init_Nature(DM,nd,nroot):
    Nature.init (DM)
    entete = nd.createElement("Execution-HyperHeuristique")
    att1 = nd.createAttribute("NombreHeuristiques")
    att1.nodeValue = str(Nature.maxH)
    att2 = nd.createAttribute("TaillePopulation")
    att2.nodeValue = str(Nature.maxP)
    entete.setAttributeNode(att1)
    entete.setAttributeNode(att2)
    premier_elem = nd.createElement("EtatPopulationInitiale")
    att1 = nd.createAttribute("AlphaIdentite")
    att1.nodeValue = Nature.actualalpha.identity
    premier_elem.setAttributeNode(att1)
    att1 = nd.createAttribute ("PrecisionMaximale")
    att1.nodeValue = str(Nature.actual_precision)
    premier_elem.setAttributeNode (att1)
    att1 = nd.createAttribute ("QualiteCumulee")
    att1.nodeValue = str(Nature.qualite)
    premier_elem.setAttributeNode (att1)
    att1 = nd.createAttribute ("Taille")
    att1.nodeValue = str(Nature.taille)
    premier_elem.setAttributeNode (att1)
    entete.appendChild(premier_elem)
    cpt = 1
    for i in range (15):
        a = time.time ()
        Nature.evolve ()
        t =  time.time () - a
        att1 = nd.createAttribute ("Iteration")
        att1.nodeValue = str (cpt)
        premier_elem.setAttributeNode (att1)
        premier_elem = nd.createElement ("EtatPopulation")
        att1 = nd.createAttribute ("AlphaIdentite")
        att1.nodeValue = str(Nature.actualalpha.identity)

        premier_elem.setAttributeNode (att1)
        att1 = nd.createAttribute ("PrecisionMaximale")
        att1.nodeValue = str(Nature.actual_precision)
        premier_elem.setAttributeNode (att1)
        att1 = nd.createAttribute ("QualiteCumulee")
        att1.nodeValue = str(Nature.evolutivite_inter)
        premier_elem.setAttributeNode (att1)
        att1 = nd.createAttribute ("Taille")
        att1.nodeValue = str(Nature.taille)
        premier_elem.setAttributeNode (att1)
        att1 = nd.createAttribute ("TempsCalcul")
        att1.nodeValue = str(t)
        premier_elem.setAttributeNode (att1)
        entete.appendChild (premier_elem)
    nroot.appendChild(entete)


def Generer_Tests_Heuristiques(data,target,nom_dataset):
    DM = Destiny()
    DM.fit(data,target)
    nd = minidom.Document ()
    nroot = nd.createElement ("Benchmarking")
    att = nd.createAttribute ("DatasetName")
    att.nodeValue = nom_dataset
    nroot.setAttributeNode (att)
    Ecrire_Dictionnaire_Heuristiques (DM , nd , nroot)
    Ecrire_Test_Heuristiques (DM , nd , nroot)
    nd.appendChild(nroot)
    f = open ("Benchmark_" + att.nodeValue + ".xml" , "w")
    nd.writexml (f , "\n" , '\t')

def Executer_Hyperheuristique(data,target,name):
    DM = Destiny ()
    DM.fit (data , target)
    nd = minidom.Document ()
    nroot = nd.createElement ("Benchmarking")
    att = nd.createAttribute ("DatasetName")
    att.nodeValue = name
    nroot.setAttributeNode (att)
    Ecrire_Init_Nature(DM,nd,nroot)
    nd.appendChild (nroot)
    f = open("Resultats_"+att.nodeValue+".xml","w")
    nd.writexml(f,"\n",'\t')

data,target = load_australian_dataset()
Generer_Tests_Heuristiques(data,target,"Australian")
Executer_Hyperheuristique(data,target,'Australian')

data,target = load_promoter_dataset()
Generer_Tests_Heuristiques(data,target,"Promoter")
Executer_Hyperheuristique(data,target,'Promoter')

data,target = load_musk_dataset()
Generer_Tests_Heuristiques(data,target,"Musk")
Executer_Hyperheuristique(data,target,'Musk')



