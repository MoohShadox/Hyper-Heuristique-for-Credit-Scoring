from Destiny import Destin as dest
from Destiny.DataSets.load_promoters_dataset import load_promoter_dataset
from Nature2 import Nature as nat
from PyQt5 import QtWidgets,uic
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class interface:



    def clear(self):
        self.DM.mesures_information.clear()
        self.DM.mesures_distance.clear()
        self.DM.mesures_dependance.clear()
        self.DM.mesures_classification.clear()
        self.DM.mesures_consistance.clear()

    def refresh(self):
        self.clear()
        i=0
        while i < self.interfac.heuristics.count():
            if self.DM.Mmesures_classification.__contains__(self.interfac.heuristics.item(i).text()):
                self.DM.mesures_classification.append(self.interfac.heuristics.item(i).text())
            if self.DM.Mmesures_consistance.__contains__(self.interfac.heuristics.item(i).text()):
                self.DM.mesures_consistance.append(self.interfac.heuristics.item(i).text())
            if self.DM.Mmesures_dependance.__contains__(self.interfac.heuristics.item(i).text()):
                self.DM.mesures_dependance.append(self.interfac.heuristics.item(i).text())
            if self.DM.Mmesures_distance.__contains__(self.interfac.heuristics.item(i).text()):
                self.DM.mesures_distance.append(self.interfac.heuristics.item(i).text())
            if self.DM.Mmesures_information.__contains__(self.interfac.heuristics.item(i).text()):
                self.DM.mesures_information.append(self.interfac.heuristics.item(i).text())
            i=i+1
        print("laa",self.DM.mesures_information)
        nat.Nature.Pstop=float(self.interfac.prob_stop.text())
        nat.Nature.Psupp=float(self.interfac.prob_supp.text())
        nat.Nature.maxA=float(self.interfac.max_attribute.text())
        nat.Nature.maxP=float(self.interfac.max_population.text())
        nat.Nature.nb_promo=float(self.interfac.nb_promotions.text())
        nat.Nature.tol_evolutivite=float(self.interfac.tol_evolutivite.text())
        nat.Nature.Tol=float(self.interfac.tol.text())
        nat.Nature.scrutin=self.interfac.ballot.currentText()
        self.DM.setSeuillage(self.interfac.treshold_type.currentText())
        if self.interfac.treshold_type.currentText()=="union_intersection":
            self.DM.setMax_iterations(int(self.interfac.nb_tresholding_union.text()))
        if self.interfac.treshold_type.currentText()=="unique":
            self.DM.setMax_iterations(int(self.interfac.nb_tresholding_unique.text()))
        if self.interfac.treshold_type.currentText() == "manual":
            self.DM.setTreshold(float(self.interfac.manual_treshold.text()))
        nat.Nature.metric=self.interfac.metrics.currentText()
        model=self.interfac.training_model.currentText()
        if model == "SVM":
            nat.Nature.modele=SVC()
        if model == "DTC":
            nat.Nature.modele=DecisionTreeClassifier()
        if model == "MLP":
            nat.Nature.modele=MLPClassifier()
        if model == "Adaboost":
            nat.Nature.modele=AdaBoostClassifier()
        if model == "KNN":
            nat.Nature.modele=KNeighborsClassifier()
        if model == "RF":
            nat.Nature.modele=RandomForestClassifier()

    def changer_scrutin(self):
        if self.interfac.ballot.currentText()=="Condorecet":
            self.interfac.tol.setEnabled(True)
        else:
            self.interfac.tol.setEnabled(False)


    def add_heuritistic(self):
        heuristic=self.interfac.add_heuristics.currentText()
        i = 0
        lebool=False
        while i < self.interfac.heuristics.count():
            if  self.interfac.heuristics.item(i).text()==heuristic:
                lebool=True
            i=i+1
        if lebool==False:
            self.interfac.heuristics.addItem(heuristic)

    def clear_heuristics(self):
        self.interfac.heuristics.clear()

    def delete_heuristic(self):
            self.interfac.heuristics.takeItem(self.interfac.heuristics.currentRow())

    def affichier_running(self):
        if(self.interfac.Running.isVisible()==False):
            self.interfac.Running.setVisible(True)
        else:
            self.interfac.Running.setVisible(False)
    def gerer_seuil(self):
        if self.interfac.treshold_type.currentText()=="union_intersection":
            self.interfac.union_treshold_label.setEnabled(True)
            self.interfac.nb_tresholding_union.setEnabled(True)
            self.interfac.unique_treshold_label.setEnabled(False)
            self.interfac.manual_treshold_label.setEnabled(False)
            self.interfac.manual_treshold.setEnabled(False)
            self.interfac.nb_tresholding_unique.setEnabled(False)

        if self.interfac.treshold_type.currentText()=="manual":
            self.interfac.union_treshold_label.setEnabled(False)
            self.interfac.nb_tresholding_union.setEnabled(False)
            self.interfac.unique_treshold_label.setEnabled(False)
            self.interfac.nb_tresholding_unique.setEnabled(False)
            self.interfac.manual_treshold_label.setEnabled(True)
            self.interfac.manual_treshold.setEnabled(True)
        if self.interfac.treshold_type.currentText()=="unique":
            self.interfac.union_treshold_label.setEnabled(False)
            self.interfac.nb_tresholding_union.setEnabled(False)
            self.interfac.unique_treshold_label.setEnabled(True)
            self.interfac.nb_tresholding_unique.setEnabled(True)
            self.interfac.manual_treshold_label.setEnabled(False)
            self.interfac.manual_treshold.setEnabled(False)

    def run(self):
        self.refresh()
        data, target = load_promoter_dataset()
        self.DM.fit(data,target)
        nat.Nature.init(self.DM)
        print("gege")
        self.interfac.iteration.setText("0")
        self.interfac.quality.setText(str(nat.Nature.actual_precision))
        for i in range(int(self.interfac.max_iteration.text())):
            nat.Nature.evolve()
            self.interfac.iteration.setText(str(i))
            self.interfac.quality.setText(str(nat.Nature.actual_precision))



    def init_interface1(self):
        self.interfac.treshold_type.addItem("manual")
        self.interfac.treshold_type.addItem("union_intersection")
        self.interfac.add_heuristics.addItems(["FScore","RST","ReliefF","FCS",'GainInformation' ,"GainRatio" , "SymetricalIncertitude" ,"MutualInformation" , "UH" , "US" , "DML","AdaBoost","RST"])
        self.interfac.metrics.addItem("accuracy")
        self.interfac.ballot.addItem("None")
        self.interfac.ballot.addItem("Condorecet")
        self.interfac.training_model.addItems(["SVM","MLP","DTC","RF","Adaboost","KNN"])
        self.interfac.prob_stop.setText(str(nat.Nature.Pstop))
        self.interfac.prob_supp.setText(str(nat.Nature.Psupp))
        self.interfac.max_attribute.setText(str(nat.Nature.maxA))
        self.interfac.max_population.setText(str(nat.Nature.maxP))
        self.interfac.nb_promotions.setText(str(nat.Nature.nb_promo))
        self.interfac.tol_evolutivite.setText(str(nat.Nature.tol_evolutivite))
        self.interfac.tol.setText(str(nat.Nature.Tol))
        self.interfac.nb_cluster.setText(str(nat.Nature.nb_cluster))
        self.interfac.max_iteration.setText(str(0))
        self.interfac.manual_treshold.setText(str(0.5))
        self.interfac.Running.setVisible(False)
        self.interfac.strategies.setColumnCount(4)
        self.interfac.strategies.setShowGrid(True)
        self.interfac.evolve_strategies.setChecked(True)


    def init_interface2(self):
        self.interfac.heuristics_button.clicked.connect(self.add_heuritistic)
        self.interfac.bake.clicked.connect(self.refresh)
        self.interfac.clear.clicked.connect(self.clear_heuristics)
        self.interfac.heuristics.addItems(self.DM.mesures_consistance)
        self.interfac.heuristics.addItems(self.DM.mesures_classification)
        self.interfac.heuristics.addItems(self.DM.mesures_dependance)
        self.interfac.heuristics.addItems(self.DM.mesures_distance)
        self.interfac.heuristics.addItems(self.DM.mesures_information)
        self.interfac.ballot.currentIndexChanged.connect(self.changer_scrutin)
        self.interfac.treshold_type.currentIndexChanged.connect(self.gerer_seuil)
        self.interfac.delete_h.clicked.connect(self.delete_heuristic)
        self.interfac.running.clicked.connect(self.affichier_running)
        self.interfac.run.clicked.connect(self.run)

    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        self.interfac = uic.loadUi("interface.ui")
        self.init_interface1()
        self.DM = dest.Destiny("manual")
        self.init_interface2()
        self.interfac.show()
        app.exec_()


dlg=interface()

