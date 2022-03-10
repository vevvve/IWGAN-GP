import numpy as np
from matplotlib import pyplot as plt
data = np.load('./8000.npy')
print(data)
# list = data.tolist()
new = np.zeros((64,64), dtype= int)
sum = 0
with open('80000.txt','w') as f:
    for i in data[0]:
        for m in range(64):
            for k in range(64):
                if i[m][k] > 0.4 or i[m][k] == 0.4:
                    new[m][k] = 1
                    sum = 1+sum
                else:
                    new[m][k] = 0
                f.write(np.str(new[m][k])+'\n')
        # plt.imshow(new)
        # plt.show()
# for i in range(64):
# image = data[1][1]
# plt.imshow(image)
# plt.show()
print(sum)
