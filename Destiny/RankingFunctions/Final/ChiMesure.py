from sklearn.feature_selection import chi2


class chi:
    def __init__(self,data,target):
        self.__data = data
        self.__target = target
        self.__scores = {}
        sc = chi2(data,target)
        for i in range(0,len(sc[0])-1):
            t = i,sc[0][i]
            self.__scores[i] = t

    def score(self,x):
        if(len(x)>1):
            return -1
        return self.__scores[x[0]]