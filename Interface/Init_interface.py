from Destiny import Destin as dest
from Destiny.DataSets.load_promoters_dataset import load_promoter_dataset
from Nature2 import Nature as nat
from PyQt5 import QtWidgets,uic
import sys

class interface:



    def clear(self):
        self.DM.mesures_information.clear()
        self.DM.mesures_distance.clear()
        self.DM.mesures_dependance.clear()
        self.DM.mesures_classification.clear()
        self.DM.mesures_consistance.clear()

    def refresh(self):
        i=0
        while i < self.interfac.heuristics.count():
            if self.DM.Mmesures_classification.__contains__(self.interfac.heuristics.item(i)):
                print('ok')
                self.DM.mesures_classification.append(self.interfac.heuristics.item(i))
            if self.DM.Mmesures_consistance.__contains__(self.interfac.heuristics.item(i)):
                print('ok')
                self.DM.mesures_consistance.append(self.interfac.heuristics.item(i))
            if self.DM.Mmesures_dependance.__contains__(self.interfac.heuristics.item(i)):
                print('ok')
                self.DM.mesures_dependance.append(self.interfac.heuristics.item(i))
            if self.DM.Mmesures_distance.__contains__(self.interfac.heuristics.item(i)):
                print('ok')
                self.DM.mesures_distance.append(self.interfac.heuristics.item(i))
            if self.DM.Mmesures_information.__contains__(self.interfac.heuristics.item(i)):
                print('ok')
                self.DM.mesures_information.append(self.interfac.heuristics.item(i))
            i=i+1

    def add_heuritistic(self):
        heuristic=self.interfac.add_heuristics.currentText()
        self.interfac.heuristics.addItem(heuristic)

    def clear_heuristics(self):
        self.interfac.heuristics.clear()

    def init_interface1(self):
        self.interfac.seuil.setVisible(False)
        self.interfac.treshold_type.addItem("manual")
        self.interfac.treshold_type.addItem("union_intersection")
        self.interfac.add_heuristics.addItems(["FScore","RST","ReliefF","FCS",'GainInformation' ,"GainRatio" , "SymetricalIncertitude" ,"MutualInformation" , "UH" , "US" , "DML","AdaBoost","RST"])
        self.interfac.metrics.addItem("accuracy")
        self.interfac.ballot.addItem("Condorecet")
        self.interfac.training_model.addItems(["SVM","MLP","DTC","RF"])
        self.interfac.prob_stop.setText(str(nat.Nature.Pstop))
        self.interfac.prob_supp.setText(str(nat.Nature.Psupp))
        self.interfac.max_attribute.setText(str(nat.Nature.maxA))
        self.interfac.max_population.setText(str(nat.Nature.maxP))
        self.interfac.nb_promotions.setText(str(nat.Nature.nb_promo))
        self.interfac.tol_evolutivite.setText(str(nat.Nature.tol_evolutivite))


    def init_interface2(self):
        self.interfac.heuristics_button.clicked.connect(self.add_heuritistic)
        self.interfac.bake.clicked.connect(self.refresh)
        self.interfac.clear.clicked.connect(self.clear_heuristics)
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        self.interfac = uic.loadUi("interface.ui")
        self.init_interface1()
        data, target = load_promoter_dataset()
        self.DM = dest.Destiny("manual")
        self.init_interface2()
        self.interfac.show()
        app.exec_()


dlg=interface()

