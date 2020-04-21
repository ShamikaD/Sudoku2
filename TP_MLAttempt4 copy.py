#all packages came with anaconda from: https://www.anaconda.com/
#pandas information from the website: https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/
#more pandas information:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html#numpy information: https://docs.scipy.org/doc/numpy/user/quickstart.html 

import pandas as pd
import numpy as np
import random, pickle
import time
start_time = time.time()
with open('trainX.pickle', 'rb') as handle:
    trainX = pickle.load(handle)
with open('testX.pickle', 'rb') as handle:
    testX = pickle.load(handle)
train = np.array([])
count = 0
for x in trainX:
    total = np.array([])
    print("working train... ", count)
    count+=1
    for i in range(128):
        total = np.concatenate((total,np.array(x[i])), axis = None)
    train = np.concatenate((train, np.array(total)), axis=0)

with open('trainX2.pickle', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)

test = np.array([])
count = 0
for x in testX:
    total = []
    print("working test... ", count)
    count+=1
    for i in range(128):
        total.extend(x[i])
    test = np.concatenate((train, np.array(total)), axis=0)

with open('testX2.pickle', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)


qerg
'''
dic = {"a": [[[1,2,3],[4,5,6],[7]]],
        "b": [[[1,2,3]],[[2]]],
        "c": [[[1,2,3],[4,5,6],[7,8,9]],[[5],[7]]]}
df = pd.DataFrame.from_dict(dic, orient='index')
dfx = pd.DataFrame()
rows = list(df.index.values)

for r in range(len(rows)):
    for c in range(len(list(df.columns.values))):
        newcol = [df.iloc[r][c]]+[rows[r]]
        dfx = dfx.append([newcol])
dfx = dfx.dropna()
y = dfx.iloc[::,-1]
x = dfx.drop(dfx.columns[-1],axis=1)

print(dfx)
print(x)
print("THISISY",y)

################################################
# all code for fixing datatset below

def letterConvert(letter):
    zeroes = [0] * 26
    num = ord(letter) - ord("a")
    zeroes[num] = 1
    return pd.Series(zeroes)


#got data from: https://catalog.data.gov/dataset/nist-handprinted-forms-and-characters-nist-special-database-19
#pickling: https://www.datacamp.com/community/tutorials/pickle-python-tutorial

with open('allLetterData.pickle', 'rb') as handle:
    openedFile = pickle.load(handle)
print ("unpickling took", (time.time() - start_time)/60, "to run")
#https://pandas.pydata.org/pandas-docs/version/0.23.1/generated/pandas.DataFrame.from_dict.html
data = pd.DataFrame.from_dict(openedFile, orient='index')
print ("converting to DF took", (time.time() - start_time)/60, "to run")
data = data.dropna(axis='columns')
allData = pd.DataFrame()
rows = list(data.index.values)
for r in range(len(rows)):
    count = 0
    for c in range(len(list(data.columns.values))):
        newRow = []
        d = [data.iloc[r][c]]
        for l in d: #this is to flatten the array
            newRow.extend(l)
        newRow.append(rows[r])
        allData = pd.concat([allData, pd.DataFrame([newRow])])
        print("working on " + rows[r]+ " "+ str(10841-count))
        count+=1
allData = allData.dropna()
y = allData.iloc[::,-1].apply(letterConvert)
x = allData.drop(allData.columns[-1],axis=1)

print("fixing the dataframe took", (time.time() - start_time)/60, "to run")
with open('allX.pickle', 'wb') as handle:
    pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('allY.pickle', 'wb') as handle:
    pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("PICKLED")

sdsc

with open('allX.pickle', 'rb') as handle:
    x = pickle.load(handle)
with open('allY.pickle', 'rb') as handle:
    y = pickle.load(handle)



print ("unpickling took ", (time.time() - start_time)/60, " to run")
#data = pd.DataFrame(openedFile)

#partitions the data into trainin and testing, and input and output
#Partition algorithm based off of: https://github.com/maddenmoore/neural-network-visualizer 

def partition(data):
    #print(data)
    allData = pd.DataFrame()
    rows = list(data.index.values)
    for r in range(len(rows)):
        count = 0
        for c in range(len(list(data.columns.values))):
            newCol = [data.iloc[r][c]]+[ord(rows[r])]
            allData = allData.append([newCol])
            print("working on " + rows[r]+ " "+ str(count))
            count+=1
    allData = allData.dropna()
    y = allData.iloc[::,-1]
    x = allData.drop(allData.columns[-1],axis=1)
    print("fixing the dataframe took", (time.time() - start_time)/60, "to run")
    numDataPts = len(x)
    numRowTrain = int(numDataPts * 0.8) #to make a split of 80% for training
    indexList = set(random.sample(list(range(numDataPts)), numRowTrain))
    trainX = pd.DataFrame()
    testX = pd.DataFrame()
    trainY = pd.DataFrame()
    testY = pd.DataFrame()
    #puts all the data in indicies that are not training into the testing data
    for i in range(numDataPts): 
        if i in indexList:
            trainX = trainX.append(x.iloc[i]) #iloc is used for indexing in pandas data frames
            trainY = trainY.append([y.iloc[i]]).astype(int)
        else:
            testX = testX.append(x.iloc[i])
            testY = testY.append([y.iloc[i]]).astype(int)
    return np.array(trainX), np.array(testX), np.array(trainY), np.array(testY)

trainX, testX, trainY, testY = partition(data)
print ("partitioning took", time.time() - start_time, "to run")

def partition(x, y):
    
    #print(data)
    allData = pd.DataFrame()
    rows = list(data.index.values)
    for r in range(len(rows)):
        count = 0
        for c in range(len(list(data.columns.values))):
            newCol = [data.iloc[r][c]]+[ord(rows[r])]
            allData = allData.append([newCol])
            print("working on " + rows[r]+ " "+ str(count))
            count+=1
    allData = allData.dropna()
    #y = allData.iloc[::,-1]
    #x = allData.drop(allData.columns[-1],axis=1)
    print("fixing the dataframe took", (time.time() - start_time)/60, "to run")
    
    numDataPts = len(x.index)
    #print(numDataPts)
    #print(type(allData))
    #print(allData.head())
    numRowTrain = int(numDataPts * 0.8) #to make a split of 80% for training
    indexList = set(random.sample(list(range(numDataPts)), numRowTrain))
    testX = pd.DataFrame()
    testY = pd.DataFrame()
    trainX = pd.DataFrame()
    trainY = pd.DataFrame()
    for i in range(numDataPts):
        print("partitioning... ", numDataPts - i) 
        if i in indexList:
            trainX = trainX.append(x.iloc[i]) #iloc is used for indexing in data frames
            trainY = trainY.append(y.iloc[i]).astype(int)
        else:
            testX = testX.append(x.iloc[i])
            testY = testY.append(y.iloc[i]).astype(int)
    return np.array(trainX), np.array(testX), np.array(trainY), np.array(testY)
    

    for i in range(numDataPts): 
        if i in indexList:
            
            testX = np.append(testX, x.iloc[i])
            testY = np.append(testY, y.iloc[i])
            x = x.drop(x.index[i]) #iloc is used for indexing in pandas data frames
            y = y.drop(y.index[i])
            i-=1
            
            train = train.append(allData.iloc[i])
        else:
            test = test.append(allData.iloc[i])
    return np.array(train.drop(allData.columns[-1],axis=1)), np.array(test.drop(test.columns[-1],axis=1)), np.array(train.iloc[::,-1]), np.array(test.iloc[::,-1]) 

trainX, testX, trainY, testY = partition(x, y)
print ("partitioning took", (time.time() - start_time)/60, "to run")
'''
with open('trainX.pickle', 'rb') as handle:
    trainX = pickle.load(handle)
with open('testX.pickle', 'rb') as handle:
    testX = pickle.load(handle)
with open('trainY.pickle', 'rb') as handle:
    trainY = pickle.load(handle)
with open('testY.pickle', 'rb') as handle:
    testY = pickle.load(handle)
print(type(trainX))

print ("unpickling took", time.time() - start_time, "to run")

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

trainingCycles = 1500 
#Trains the net with the training set
count = 0
for x in range(trainingCycles):
    net.forward()
    net.back()
    print ("training... "+ str(1500 - count))
    print ("training is taking ", (time.time() - start_time)/60, " mins per cycle")
    count+=1
print ("training took ", (time.time() - start_time)/60, " to run")
		
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
print ("last cycle took", time.time() - start_time, "to run")