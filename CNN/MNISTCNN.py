# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct

# -----------------all the necessary data to read the data file is given at http://yann.lecun.com/exdb/mnist/
Train_Data = open('../../Data/MNIST/train-images-idx3-ubyte', 'rb')
Train_Labels = open('../../Data/MNIST/train-labels-idx1-ubyte', 'rb')

# -----------------Skip the magic number
Train_Data.read(4)
Train_Labels.read(4)

# -----------------Get Dimensions
imageCount = struct.unpack('>i', Train_Data.read(4))[0]
imageSize =  [struct.unpack('>i', Train_Data.read(4))[0], struct.unpack('>i', Train_Data.read(4))[0]]

LabelCount = struct.unpack('>i', Train_Labels.read(4))[0]

# -----------------Create a numpy array with all the images
Train_X = np.zeros([imageCount, imageSize[0], imageSize[1]], dtype=float)

# print(np.prod(imageSize))

for i in range(imageCount):
    Train_X[i] = np.frombuffer(Train_Data.read(np.prod(imageSize)), dtype=np.uint8).reshape(imageSize)

# first_image = np.frombuffer(Train_Data.read(28* 28), dtype=np.uint8).reshape(28, 28)

Labels = np.frombuffer(Train_Labels.read(), dtype=np.uint8)


# plt.imshow(Train_X[0])
# plt.show()


