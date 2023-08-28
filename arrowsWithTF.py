import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from scipy import signal

from arrowRecognitonShallow import data

numExamples = 5000
numHiddenNodes = 10

dataGet = data(numExamples)

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


edgesLin = edges.reshape((numExamples, -1)).T

# outline the computational graph up here
x = tf.placeholder(tf.float32,edgesLin.shape)
y = tf.placeholder(tf.float32,dataGet.y.shape)

print(numHiddenNodes,dataGet.xRes*dataGet.yRes)

w1 = tf.Variable(tf.random_normal((numHiddenNodes,dataGet.xRes*dataGet.yRes)))
w2 = tf.Variable(tf.random_normal((1,numHiddenNodes)))

b1 = tf.Variable(tf.random_normal((numHiddenNodes,1)))
b2 = tf.Variable(tf.random_normal((1,1)))
                 
# linear classify then ReLU
h = tf.maximum(tf.add(tf.matmul(w1,x),b1),0)
h2 = tf.add(tf.matmul(w2,h),b2)
y_pred = 2*np.pi/(1+tf.exp(-0.00001*h2))
loss = tf.losses.mean_squared_error(y_pred,y)

#grad_w1, grad_w2, grad_b1, grad_b2 = tf.gradients(loss, [w1,w2,b1,b2])
#learningRate = 1e-2

#new_w1 = w1.assign(w1 - learningRate * grad_w1)
#new_w2 = w2.assign(w2 - learningRate * grad_w2)

#new_b1 = b1.assign(b1 - learningRate * grad_b1)
#new_b2 = b2.assign(b2 - learningRate * grad_b2)

#updates = tf.group(new_w1, new_w2, new_b1, new_b2)

optimizer = tf.train.GradientDescentOptimizer(1e-1)
updates = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {x: edgesLin, y: dataGet.y}
    losses = []

    i = 0

    while True:
        try:
            loss_val , _ = sess.run([loss, updates],feed_dict=values)
            i += 1

            if i%100 == 0:
                print(loss_val)

        except KeyboardInterrupt:

            w1, w2, b1, b2 = sess.run([w1, w2, b1, b2],feed_dict=values)

            np.save("w1.npy",w1)
            np.save("w2.npy",w2)

            np.save("b1.npy",b1)
            np.save("b2.npy",b2)
            quit()

