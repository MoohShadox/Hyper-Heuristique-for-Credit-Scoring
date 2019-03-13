import Nature as nat
import random
#natu=nat.Nature()
#print(natu.actualalpha)
#natu.evolve()
#print(natu.actualalpha)

nat.Nature.init()
#print(nat.Nature.actualalpha.identity)
nat.Nature.evolve()
print([s.identity for s in nat.Nature.population])
print(random.random())
