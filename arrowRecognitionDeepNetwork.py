import numpy as np
import random
import pygame

import matplotlib.pyplot as plt
from scipy import signal

import numpy.linalg as la
import math

class Arrow():

    def __init__(self):
        self.arrowShape = np.array([[0, 100], [0, 200], [200, 200], [200, 300], [300, 150], [200, 0], [200, 100]])

        self.arrowShape = self.arrowShape - np.array([150,150])

        self.arrowShape = np.multiply(1/300,self.arrowShape)

        # set constant width 

        self.arrowShape[:,1] *= 10


    def plot(self,angle,radius,centre):

        newPoints = []

        rotMat = np.array([[math.cos(angle),-math.sin(angle)],
                        [math.sin(angle),math.cos(angle)]])
        furthestPoint = 0

        """for point in self.arrowShape:
            shapeLength = la.norm(point)
            if shapeLength > la.norm(furthestPoint):
                furthestPoint = point"""

        scaleFac = radius
        for i in range(len(self.arrowShape)):
            newPoints.append(np.array([self.arrowShape[i,0]*scaleFac, self.arrowShape[i,1]]))
            newPoints[i] = np.matmul(rotMat,newPoints[i])
            newPoints[i] += centre

        return newPoints

    def plotVec(self,pos,vec,scaleFac):
        angle = np.arctan2(vec[1],vec[0])
        length = scaleFac*la.norm(vec)
        points = self.plot(angle,length,pos)
        return points

class data():
    def __init__(self,numExamples):
        pygame.init()

        self.xRes = 41
        self.yRes = 41
        self.depth = 3

        self.picMiddle = np.array([self.xRes/2, self.yRes/2])

        self.screen = pygame.display.set_mode((self.xRes,self.yRes))

        self.radius = np.minimum(self.xRes/2,self.yRes/2)-2

        print(self.radius)

        self.x = np.zeros((numExamples,self.xRes,self.yRes,self.depth))
        self.y = np.zeros((numExamples))

        arrow = Arrow()

        for i in range(numExamples):
            direction = np.random.rand()*2*np.pi
            self.y[i] = direction
            arrowPoints = arrow.plot(direction,20,self.picMiddle)
            self.screen.fill((0,0,0))
            pygame.draw.polygon(self.screen, (255, 0, 0),arrowPoints)
            self.x[i,:] = pygame.surfarray.array3d(self.screen)#.flatten(

def linRegresOne(w,x,b):
    return np.add(w.dot(x),b)

def lossFunction(targetVec,actualVec):
    return (1/len(targetVec))*np.sum(np.square(np.subtract(targetVec,actualVec)))

def sigmoid(x):
    return 2*np.pi/(1+np.exp(-0.0001*x))

def ReLU(x):
    return np.maximum(0, x)

#def backPass(x,y,y_g):
#    return (2/len(x))*(y-y_g).dot(x.T)

numExamples = 1000
dataGet = data(numExamples)
m = len(dataGet.x)

learnRate = 0.00001
layer1Nodes = 16
layer2Nodes = 10

## prep data
grayIms = np.dot(dataGet.x[...,:3], [0.299, 0.587, 0.114]) # take up to the 3rd index

xKernel = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0,-1]])

yKernel = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2,-1]])

print(grayIms.shape)

edges = np.zeros(grayIms.shape)

for i in range(len(grayIms)):

    Gx = signal.convolve2d(grayIms[i,:],xKernel, mode='same', boundary='symm')
    Gy = signal.convolve2d(grayIms[i,:],yKernel, mode='same', boundary='symm')

    edges[i,:] = np.sqrt(np.square(Gx)+np.square(Gy))


edgesLin = edges.reshape((numExamples, -1))

w1 = np.random.rand(layer1Nodes,dataGet.xRes*dataGet.yRes)-0.5
b1 = np.random.rand(layer1Nodes,1)

w2 = np.random.rand(layer2Nodes,layer1Nodes)
b2 = np.random.rand(layer2Nodes,1)

w3 = np.random.rand(1,layer2Nodes)
b3 = np.random.rand(1,1)


if __name__ == "__main__":

    i = 0
    while True:
        try:
            x_2 = linRegresOne(w1,edgesLin.T,b1)

            x_3 = ReLU(x_2)

            x_4 = linRegresOne(w2,x_3,b2)

            x_5 = ReLU(x_4)

            x_6 = linRegresOne(w3,x_5,b3)

            x_7 = sigmoid(x_6)

            sigDiv = sigmoid(x_6)*(1-sigmoid(x_6))

            postCostDiv = -(2/m)*np.multiply(sigDiv,(dataGet.y-x_7))

            dW3 = postCostDiv.dot(x_5.T)
            dW2 = w3.T.dot(postCostDiv).dot(x_3.T)
            dW1 = w2.T.dot(w3.T.dot(postCostDiv)).dot(edgesLin)

            db3 = np.sum(postCostDiv)
            db2= np.sum(w3.T.dot(postCostDiv))
            db1 = np.sum(w2.T.dot(w3.T.dot(postCostDiv)))

            w1 = np.add(w1,(learnRate/m)*dW1)
            w2 = np.add(w2,(learnRate/m)*dW2)
            w3 = np.add(w3,(learnRate/m)*dW3)

            b1 = np.add(b1,(learnRate/m)*db1)
            b2 = np.add(b2,(learnRate/m)*db2)
            b3 = np.add(b3,(learnRate/m)*db3)

            if i % 100 == 0:
                print(f"Run: {i} Loss {lossFunction(dataGet.y,x_7)}")
            i += 1

        except KeyboardInterrupt:
            np.save("w1.npy",w1)
            np.save("w2.npy",w2)
            np.save("w3.npy",w3)

            np.save("b1.npy",b1)
            np.save("b2.npy",b2)
            np.save("b3.npy",b3)
            quit()


    np.save("w1.npy",w1)
    np.save("w2.npy",w2)
    np.save("w3.npy",w3)

    np.save("b1.npy",b1)
    np.save("b2.npy",b2)
    np.save("b3.npy",b3)

    # test 

    testData = data(1000)

    grayIms = np.dot(testData.x[...,:3], [0.299, 0.587, 0.114]) # take up to the 3rd index

    for i in range(len(grayIms)):

        Gx = signal.convolve2d(grayIms[i,:],xKernel, mode='same', boundary='symm')
        Gy = signal.convolve2d(grayIms[i,:],yKernel, mode='same', boundary='symm')

        edges[i,:] = np.sqrt(np.square(Gx)+np.square(Gy))

    edgesLin = edges.reshape((numExamples, -1))

    x_1 = linRegresOne(w1,edgesLin.T,b1)
    x_2 = ReLU(x_1)
    x_3 = linRegresOne(w2,x_2,b2)
    y_g = sigmoid(x_3)


    correct = 0

    for i in range(len(y_g.T)):
        if y_g[i] - testData.y[i] < np.pi/6:
            correct += 1

    print(f"{correct*100/1000} % success")
