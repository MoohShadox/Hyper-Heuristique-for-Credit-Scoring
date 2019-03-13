import Nature as nat
import re

class Genome:

    def __init__(self):
        self.identity=""
        self.incarnation=[]
        self.isvalide=0

    def incarner(self):
        exp = "(\d[h\d]+)/"
        genes=re.findall(exp,self.identity)

        for g in genes:
            nbattribute = int(g[0])
            flist=[]
            i=2
            while(i<len(g)):
                flist.append(g[i])
                i=i+2
            self.incarnation.append((g,None,None))
        return None