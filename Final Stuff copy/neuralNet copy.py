#all packages came with anaconda from: https://www.anaconda.com/
#pandas information from the website: https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/
#more pandas information:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html#numpy information: https://docs.scipy.org/doc/numpy/user/quickstart.html 

import pandas as pd
import numpy as np
import random, pickle
import time
start_time = time.time()

#neural network based on: from website: https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7
#sigmoid function: used to normalize values
#Information I used to understand the math: https://en.wikipedia.org/wiki/Sigmoid_function 
def sig(n):
    return 1/(1 + np.exp(-n))

#derivative of the sig function (just take the derivative of the sigmoid function)
def sigDer(n): 
    return n * (1 - n)

#turns a data into probability distrubutions
#Information I used to understand the math: https://towardsdatascience.com/softmax-function-simplified-714068bf8156
def softmax(n):
    nMax = np.max(n, axis=1, keepdims=True)
    eExpression = np.exp(n - nMax)
    eExpressionTotal = np.sum(eExpression, axis=1, keepdims=True)
    return eExpression/eExpressionTotal

#this is the loss function that calculated the loss at each layer during backpropogation
#here, cross entropy is usually used. Explination here: https://machinelearningmastery.com/cross-entropy-for-machine-learning/
#I used general, linear error (like used in chem)
def loss(prediction, real):
    numRowsReal = real.shape[0]
    return (prediction - real)/numRowsReal

class NeuralNet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.learningRate = 0.5
        amountInputs = x.shape[1] 
        amountOutputs = y.shape[1]
        #neuron amount calculated by these principles: https://www.heatonresearch.com/2017/06/01/hidden-layers.html
        neurons = 26*128#*128 #amountInputs * amountOutputs 
        #start with semirandom numbers for weights and biases
        self.w1 = np.random.randn(amountInputs, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, amountOutputs)
        self.b3 = np.zeros((1, amountOutputs))

    def forward(self):
        self.layer1 = sig(np.dot(self.x, self.w1) + self.b1)
        self.layer2 = sig(np.dot(self.layer1, self.w2) + self.b2)
        self.layer3 = softmax(np.dot(self.layer2, self.w3) + self.b3) #soft max takes vectors and turns them into probablity distributions
        
    def back(self):
        changeLayer3 = loss(self.layer3, self.y) # calculate the loss here to see how much to adjust by
        changeLayer2 = np.dot(changeLayer3, self.w3.T) * sigDer(self.layer2) 
        changeLayer1 = np.dot(changeLayer2, self.w2.T) * sigDer(self.layer1) 
        #after calculating loss at each level, go and change all the weights and biases
        self.w3 -= self.learningRate*np.dot(self.layer2.T, changeLayer3)
        self.b3 -= self.learningRate*np.sum(changeLayer3)
        self.w2 -= self.learningRate*np.dot(self.layer1.T, changeLayer2)
        self.b2 -= self.learningRate*np.sum(changeLayer2)
        self.w1 -= self.learningRate*np.dot(self.x.T, changeLayer1)
        self.b1 -= self.learningRate*np.sum(changeLayer1)

    #predicts from the testing data set by going forward in the NN
    #used: to get the accuracy
    def predict(self, data):
        self.x = data
        self.forward()
        mostProbableVal = self.layer3.argmax()
        return mostProbableVal
'''
with open('trainX3.pickle', 'rb') as handle:
    trainX = pickle.load(handle)
with open('testX3.pickle', 'rb') as handle:
    testX = pickle.load(handle)
with open('trainY.pickle', 'rb') as handle:
    trainY = pickle.load(handle)
with open('testY.pickle', 'rb') as handle:
    testY = pickle.load(handle)

#object of NeuralNet class
net = NeuralNet(trainX, trainY)

trainingCycles = 1500 
#Trains the net with the training set
count = 0

#returns the accuracy of the network
#if the prediction is the same as the biggest vaule in y, accuracy goes up by 1
#used: to get the accuracy of the network for the training data and the testing data
def accuracy(x, y): 
    accuracy = 0
    for i in range(len(x)):
        if net.predict(x[i]) == np.argmax(y[i]):
            accuracy += 1
    return (accuracy/len(x))*100 #returns the accuracy as a percentage of the data set accurately predicted
	
for x in range(trainingCycles):
    net.forward()
    net.back()
    print ("training... "+ str(1500 - count))
    print ("training is taking ", (time.time() - start_time)/60, " mins per cycle")
    count+=1
print ("training took ", (time.time() - start_time)/60, " to run")
print("Training accuracy : ", accuracy(trainX, trainY))
print("Testing accuracy : ", accuracy(testX, testY)) 
print ("last cycle took", time.time() - start_time, "to run")

with open('neuralNet.pickle', 'wb') as handle:
    pickle.dump(net, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
from imageToTestData import getAllLetters
words  = getAllLetters('hello.png')
print('here 1')
with open('neuralNet128Neuron.pickle', 'rb') as handle:
    net = pickle.load(handle)
print('here 2')
predWords = ""
for line in words:
    for letter in line:
        predLetter = net.predict(letter)
        print(predLetter)
        predLetAlpha = chr(ord("a")+predLetter)
        predWords+=predLetAlpha
    predWords+="\n"
print(predWords)