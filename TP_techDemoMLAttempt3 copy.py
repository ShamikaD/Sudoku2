#all packages came with anaconda from: https://www.anaconda.com/
#pandas information from the website: https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/
#more pandas information:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html#numpy information: https://docs.scipy.org/doc/numpy/user/quickstart.html 

import pandas as pd
import numpy as np
import random

#got data from: http://archive.ics.uci.edu/ml/datasets/Iris 
data = pd.DataFrame(pd.read_csv("iris.csv"))
data = data.append(data)

def flowerConversion(flower):
    if flower == "Iris-setosa":
        return pd.Series([1,0,0])
    elif flower == "Iris-versicolor":
        return pd.Series([0,1,0])
    elif flower == "Iris-virginica":
        return pd.Series([0,0,1])

#partitions the data into trainin and testing, and input and output
#Partition algorithm based off of: https://github.com/maddenmoore/neural-network-visualizer 
def partition(data):
    data.columns = ['A', "B", "C", "D", "Y"]
    x = data.drop(['Y'], axis = 1)
    y = data["Y"].apply(flowerConversion)
    print (x)
    numDataPts = len(x)
    numRowTrain = int(numDataPts * 0.7) #to make a split of 70% for training
    indexList = set(random.sample(list(range(numDataPts)), numRowTrain))
    trainX = pd.DataFrame()
    testX = pd.DataFrame() 
    trainY = pd.DataFrame()
    testY = pd.DataFrame()
    #puts all the data in indicies that are not training into the testing data
    for i in range(numDataPts): 
        if i in indexList:
            trainX = trainX.append(x.iloc[i]) #iloc is used for indexing in pandas data frames
            trainY = trainY.append(y.iloc[i]).astype(int)
        else:
            testX = testX.append(x.iloc[i])
            testY = testY.append(y.iloc[i]).astype(int)
    return np.array(trainX), np.array(testX), np.array(trainY), np.array(testY)

trainX, testX, trainY, testY = partition(data)
#print(testX, trainY)

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
        amountInputs = x.shape[1]
        amountOutputs = y.shape[1]
        #neuron amount calculated by these principles: https://www.heatonresearch.com/2017/06/01/hidden-layers.html
        neurons = amountInputs * amountOutputs 
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
        self.w3 -= np.dot(self.layer2.T, changeLayer3)
        self.b3 -= np.sum(changeLayer3)
        self.w2 -= np.dot(self.layer1.T, changeLayer2)
        self.b2 -= np.sum(changeLayer2)
        self.w1 -= np.dot(self.x.T, changeLayer1)
        self.b1 -= np.sum(changeLayer1)

    #predicts from the testing data set by going forward in the NN
    #used: to get the accuracy
    def predict(self, data):
        self.x = data
        self.forward()
        mostProbableVal = self.layer3.argmax()
        return mostProbableVal
			
#object of NeuralNet class
net = NeuralNet(trainX, trainY)

trainingCycles = 3000 
#Trains the net with the training set
for x in range(trainingCycles):
    net.forward()
    net.back()
		
#returns the accuracy of the network
#if the prediction is the same as the biggest vaule in y, accuracy goes up by 1
#used: to get the accuracy of the network for the training data and the testing data
def accuracy(x, y): 
    accuracy = 0
    for i in range(len(x)):
        if net.predict(x[i]) == np.argmax(y[i]):
            accuracy += 1
    return (accuracy/len(x))*100 #returns the accuracy as a percentage of the data set accurately predicted
	
print("Training accuracy : ", accuracy(trainX, trainY))
print("Testing accuracy : ", accuracy(testX, testY)) 
#print(net.w1)