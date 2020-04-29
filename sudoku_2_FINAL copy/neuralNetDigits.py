#pandas, numpy, and matplotlib all came with anacoda, which I downloaded
# from here: https://docs.anaconda.com/anaconda/install/

#I taught myself pandas from: 
#https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/
#The rest of the pandas information that I learned is from: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
#I taught myself matplotlib from: https://matplotlib.org/users/pyplot_tutorial.html
#the rest of the matplotlib information is from: https://matplotlib.org/tutorials/index.html#introductory
#I taught myself numpy from: https://docs.scipy.org/doc/numpy/user/quickstart.html
#the rest of the numpy information is from: https://docs.scipy.org/doc/numpy/reference/
import pandas as pd
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
#This is my own function that gets the image data to put through the neural network
from imageToBoard import forNet
#The dataset that I'm using to train is from :
#https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html 
from sklearn.datasets import load_digits

data = load_digits()

#Partition algorithm sort of based off of: https://github.com/maddenmoore/neural-network-visualizer 
def partition(data):
    x = pd.DataFrame(data.data)
    y = data.target
    y = pd.get_dummies(y) #converts all the digits into one turned on binary in a row of 10 binary digits
    #partitions the data into trainin and testing, and input and output
    numDataPts = x.shape[0]
    numRowTrain = int(numDataPts * 0.70) #to make a split of 70% for training
    indexList = set(random.sample(list(range(numDataPts)), numRowTrain))
    trainX = pd.DataFrame()
    testX = pd.DataFrame() 
    trainY = pd.DataFrame()
    testY = pd.DataFrame()
    #puts all the data in indicies that are not training into the testing data
    #and puts the training indicies into the training data
    for i in range(numDataPts): 
        if i in indexList:
            trainX = trainX.append(x.iloc[i]) #iloc is used for indexing in pandas data frames
            trainY = trainY.append(y.iloc[i]).astype(int)
        else:
            testX = testX.append(x.iloc[i])
            testY = testY.append(y.iloc[i]).astype(int)
    return np.array(trainX), np.array(testX), np.array(trainY), np.array(testY)

trainX, testX, trainY, testY = partition(data)

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

#neural network based on: https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7
class NeuralNet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        amountInputs = x.shape[1] 
        amountOutputs = y.shape[1]
        #neuron amount calculated by these principles: https://www.heatonresearch.com/2017/06/01/hidden-layers.html
        neurons = (amountInputs*2)//3 + amountOutputs 
        #start with semirandom numbers for weights and biases
        self.w1 = np.random.randn(amountInputs, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, amountOutputs)
        self.b3 = np.zeros((1, amountOutputs))

    #forward propogation is done by taking the dot product of the weights adding the biases
    #this is basically like y=mx+b where m is the weight, x is the input, and b is the bias
    #so, we try to find the best weights ad biases so that every x put in the equation gets the correct y
    def forward(self):
        self.layer1 = sig(np.dot(self.x, self.w1) + self.b1)
        self.layer2 = sig(np.dot(self.layer1, self.w2) + self.b2)
        self.layer3 = softmax(np.dot(self.layer2, self.w3) + self.b3) #soft max takes vectors and turns them into probablity distributions
        
    #back propogation calculates the loss on each layer of the network and adjusts the wrights and biases accordingly
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

#trains the network and returns the result from user input (image to 2D list)
#and converts the results of the neural network into a sudoku board
def runNet(userInput):
    #creating a neural network
    net = NeuralNet(trainX/16, trainY) #divide by 16 because 16 is the largest number present in the images
    #training the neural network
    count = 0
    trainingCycles = 2000 
    for x in range(trainingCycles):
        net.forward()
        net.back()
        print ("training... "+ str(2000 - count))
        count +=1
    #getting your game board
    hand = forNet(userInput)
    board = []
    for line in hand:
        ln = []
        for num in line:
            ln.append(net.predict(num))
        board.append(ln)
    return board