import numpy as np
from PIL import Image
import os
datasource = r'./test'
savadir = r'./data'
i=0
for file in os.listdir(datasource):
    dataset = [np.array(Image.open(datasource+'/'+file+'/'+images)) for images in os.listdir(datasource+'/'+file)]
    dataset = np.array(dataset)
    # np.save(savadir+'/'+'data%d'%(i),dataset)
    print(dataset)
    print(dataset.shape)
