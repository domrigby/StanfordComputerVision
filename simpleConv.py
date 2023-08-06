import numpy as np
import random
import pygame

import matplotlib.pyplot as plt
from scipy import signal

class data():
    def __init__(self,numExamples):
        pygame.init()

        self.xRes = 21
        self.yRes = 21
        self.depth = 3

        self.screen = pygame.display.set_mode((self.xRes,self.yRes))

        self.x = np.zeros((numExamples,self.xRes,self.yRes,self.depth))
        self.y = np.zeros((3,numExamples))

        for i in range(numExamples):
            num = np.random.rand()
            if num < 0.33:
                self.x[i,:] = self.__drawCircle()
                self.y[0,i] = 1
            elif 0.33 < num <= 0.66:
                self.x[i,:] = self.__drawSquare()
                self.y[1,i] = 1
            else:
                self.x[i,:] = self.__drawTriangle()
                self.y[2,i] = 1


    def __drawCircle(self):
        self.screen.fill((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        pygame.draw.circle(self.screen,(random.randint(0,255),random.randint(0,255),random.randint(0,255)),(10,10),8)
        return pygame.surfarray.array3d(self.screen)#.flatten()

    def __drawSquare(self):
        self.screen.fill((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        pygame.draw.rect(self.screen, (random.randint(0,255),random.randint(0,255),random.randint(0,255)), pygame.Rect(3, 3, 14, 14))
        return pygame.surfarray.array3d(self.screen)#.flatten()
    
    def __drawTriangle(self):
        self.screen.fill((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        pygame.draw.polygon(self.screen, (random.randint(0,255),random.randint(0,255),random.randint(0,255)), ((3,3), (3,18), (18,11)))
        return pygame.surfarray.array3d(self.screen)#.flatten()
    
    def __drawScreen(self):
        plt.imshow(pygame.surfarray.array3d(self.screen))
        plt.show()

def linRegresOne(w,x,b):
    return np.add(w.dot(x),b)

def lossFunction(targetVec,actualVec):
    return (1/targetVec.shape[1])*np.sum(np.square(np.subtract(targetVec,actualVec)))

def sigmoid(x):
    return 1/(1+np.exp(-0.0001*x))

def ReLU(x):
    return np.maximum(0, x)

#def backPass(x,y,y_g):
#    return (2/len(x))*(y-y_g).dot(x.T)

numExamples = 1000
dataGet = data(numExamples)
m = len(dataGet.x)

learnRate = 0.01
numHiddenNodes = 10

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

w1 = np.random.rand(numHiddenNodes,dataGet.xRes*dataGet.yRes)
b1 = np.random.rand(numHiddenNodes,1)

w2 = np.random.rand(3,numHiddenNodes)
b2 = np.random.rand(3,1)


if __name__ == "__main__":

    for i in range(10000):

        x_2 = linRegresOne(w1,edgesLin.T,b1)

        x_3 = ReLU(x_2)

        x_4 = linRegresOne(w2,x_3,b2)

        x_5 = sigmoid(x_4)

        sigDiv = sigmoid(x_4)*(1-sigmoid(x_4))

        postCostDiv = -(2/m)*np.multiply(sigDiv,(dataGet.y-x_5))

        dW2 = postCostDiv.dot(x_3.T)
        dW1 = w2.T.dot(postCostDiv).dot(edgesLin)

        db2 = np.sum(postCostDiv)
        db1 = np.sum(w2.T.dot(postCostDiv))


        w1 = np.add(w1,-(learnRate/m)*dW1)
        w2 = np.add(w2,-(learnRate/m)*dW2)

        b1 = np.add(b1,-(learnRate/m)*db1)
        b2 = np.add(b2,-(learnRate/m)*db2)

        if i % 100 == 0:
            print(f"Run: {i} Loss {lossFunction(dataGet.y,x_5)}")


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
        if y_g.T[i].argmax() == testData.y.T[i].argmax():
            correct += 1

    print(f"{correct*100/1000} % success")

    np.save("w1.npy",w1)
    np.save("w2.npy",w2)

    np.save("b1.npy",b1)
    np.save("b2.npy",b2)

    rChannel = np.reshape(w1[0,0:121],(11,11))

    plt.imshow(rChannel)
    plt.show()

