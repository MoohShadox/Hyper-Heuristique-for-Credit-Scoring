from sklearn.feature_selection import mutual_info_classif

class chi:
    def __init__(self,data,target):
        self.__data = data
        self.__target = target
        self.__scores = mutual_info_classif(data,target)

    def get_scores(self):
        return self.__scores