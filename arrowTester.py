import arrowRecognitonShallow

from scipy import signal
import numpy as np

import matplotlib.pyplot as plt

xKernel = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0,-1]])

yKernel = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2,-1]])

testData = arrowRecognitonShallow.data(1000)

testData.x = testData.x # + 0.5*np.random.rand(1000,41,41,3)

grayIms = np.dot(testData.x[...,:3], [0.299, 0.587, 0.114]) # take up to the 3rd index

edges = np.zeros(grayIms.shape)

for i in range(len(grayIms)):

    Gx = signal.convolve2d(grayIms[i,:],xKernel, mode='same', boundary='symm')
    Gy = signal.convolve2d(grayIms[i,:],yKernel, mode='same', boundary='symm')

    edges[i,:] = np.sqrt(np.square(Gx)+np.square(Gy))

edgesLin = edges.reshape((len(edges), -1))

"""w1 = np.load("GoodWeightForArrow/w1Good.npy")
w2 = np.load("GoodWeightForArrow/w2Good.npy")

b1 = np.load("GoodWeightForArrow/b1Good.npy")
b2 = np.load("GoodWeightForArrow/b2Good.npy")"""

w1 = np.load("w1.npy")
w2 = np.load("w2.npy")

b1 = np.load("b1.npy")
b2 = np.load("b2.npy")

x_1 = arrowRecognitonShallow.linRegresOne(w1,edgesLin.T,b1)
x_2 = arrowRecognitonShallow.ReLU(x_1)
x_3 = arrowRecognitonShallow.linRegresOne(w2,x_2,b2)
y_g = arrowRecognitonShallow.sigmoid(x_3)

error = np.sqrt(np.square((y_g-testData.y)/2*np.pi))
counts, bins = np.histogram(error)
plt.hist(bins[:-1], bins, weights=counts)
plt.show()

correct = 0

for i in range(len(y_g.T)):
    plt.imshow(testData.x[i,:])
    plt.title(f"Algo prediction: {360*(y_g.T[i]/(2*np.pi))-90} Actual {360*(testData.y[i]/(2*np.pi))-90}")
    plt.show()

print(f"{correct*100/1000} % success")