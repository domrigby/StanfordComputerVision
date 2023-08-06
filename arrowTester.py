import arrowRecognition

from scipy import signal
import numpy as np

import matplotlib.pyplot as plt

xKernel = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0,-1]])

yKernel = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2,-1]])

testData = arrowRecognition.data(1000)

grayIms = np.dot(testData.x[...,:3], [0.299, 0.587, 0.114]) # take up to the 3rd index

edges = np.zeros(grayIms.shape)

for i in range(len(grayIms)):

    Gx = signal.convolve2d(grayIms[i,:],xKernel, mode='same', boundary='symm')
    Gy = signal.convolve2d(grayIms[i,:],yKernel, mode='same', boundary='symm')

    edges[i,:] = np.sqrt(np.square(Gx)+np.square(Gy))

edgesLin = edges.reshape((len(edges), -1))

w1 = np.load("w1.npy")
w2 = np.load("w2.npy")

b1 = np.load("b1.npy")
b2 = np.load("b2.npy")

x_1 = arrowRecognition.linRegresOne(w1,edgesLin.T,b1)
x_2 = arrowRecognition.ReLU(x_1)
x_3 = arrowRecognition.linRegresOne(w2,x_2,b2)
y_g = arrowRecognition.sigmoid(x_3)


correct = 0
print(360*(y_g.T[i]/(2*np.pi)))

for i in range(len(y_g.T)):
    plt.imshow(testData.x[i,:])
    print(y_g.T[i])
    plt.title(f"Algo prediction: {360*(y_g.T[i]/(2*np.pi))-90} Actual {360*(testData.y[i]/(2*np.pi))-90}")
    plt.show()

print(f"{correct*100/1000} % success")