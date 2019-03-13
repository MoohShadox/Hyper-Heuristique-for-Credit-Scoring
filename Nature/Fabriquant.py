import operator
import Genome as gn
import re
import Destiny as dest
class Fabriquant:
    def __init__(self,GN):
        self.listbuffer=[]
        exp = "(\d[h\d]+)/"
        self.attlen= GN.identity[0]
        self.recette=re.findall(exp,GN.identity)
        self.genome=gn.Genome()
        exp2="h\d"
        hierlist=[]
        for stg in self.recette:
            gene=re.findall(exp2,stg)
            for st in gene:
                hierlist.append(dest.Destiny.gethierarchie(self.attlen,int(st[1])))
            vlist=[]
            for k in hierlist:
                sorted(k,key=operator.itemgetter(0))
            for i in range(len(hierlist[0])):
                k=(hierlist[0][i][0],0)
                for j in range(len(hierlist)):
                    if(hierlist[i][j][1]>0):
                        k=(hierlist[i][j][0],k+(hierlist[i][j][1])/len(hierlist))
                    else:
                        k = (hierlist[i][j][0], -1*len(hierlist))
                vlist.append(k)
            sorted(vlist,key=operator.itemgetter(1))
            for g in range(len(vlist)):
                if(intersect(self.listbuffer,vlist[g][0])==[]):
                    self.recette.append((st,vlist[g][0],vlist[g][1]))
                    g=len(vlist)+1


    def bourrage(self):
        bourlist=[]
        for k in range(len(self.recette)):
            if(self.recette[k][2]>0):
                bourlist.append(self.recette[k])
            else:
                k=len(self.recette)+1





def intersect(a, b):
    return list(set(a) & set(b))
