import numpy as np
data = np.load('data2.npy')
with open('data2.txt','a') as f:
    f.write('data'+'\n'+'1'+'\n'+'facies')
    for i in data:
        for l in i:
            for m in l:
                f.write(str(m)+'\n')