from sklearn.feature_selection import f_classif


class FScore:
    def __init__(self,data,target):
        self.__data = data
        self.__target = target
        self.__scores = {}
        sc = f_classif(data,target)
        cpt = 0
        for i in sc[0]:
            tup = cpt,i
            self.__scores[cpt] = tup
            cpt = cpt + 1

    def score(self, x):
        if (len(x) > 1):
            score = -1
        else:
            score = self.__scores[x[0]]
        return score





