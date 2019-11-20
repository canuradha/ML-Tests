# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import struct

# all the necessary data to read the data file is given at http://yann.lecun.com/exdb/mnist/
Train_Data = open('../../Data/MNIST/train-images-idx3-ubyte', 'rb')

# Skip the magic number
Train_Data.read(4)

# get Dimensions
imageCount = struct.unpack('>i', Train_Data.read(4))[0]
imageSize =  struct.unpack('>i', Train_Data.read(4))[0] * struct.unpack('>i', Train_Data.read(4))[0] 

first_image = np.frombuffer(Train_Data.read(imageSize), dtype=np.uint8)
first_image = first_image.reshape(28, 28)

print(first_image)

plt.imshow(first_image)
plt.show()


