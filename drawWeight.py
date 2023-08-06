import numpy as np
import matplotlib.pyplot as plt

w1 = np.load("w1.npy")
w2 = np.load("w2.npy")

b1 = np.load("b1.npy")
b2 = np.load("b2.npy")


for x in w1:
    im = x.reshape(21,21)
    plt.imshow(im)
    plt.show()