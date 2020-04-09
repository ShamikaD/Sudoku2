#import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import pandas as pd
#from numpy import pi

#tutorial from https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

#input layer = x (not counted when conting number of layers in NN)
#amount of hidden layers is arbitrary
#output layer = y_hat
#w = weights
#b = biases
#sigma (lowercase) = activation function (turns into 1 or 0) --> using sigmoid
'''
class NeuralNet:
    def __init__(self, training, y):
        self.input      = training
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(y.shape)


#directly coppied from website below
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


#attempt 3:

# Generate a dataset and plot it
#np.random.seed(0)
#X, y = sklearn.datasets.make_moons(200, noise=0.20)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

dataSetDF = pd.read_csv("dataset.csv", index_col = 0)
descript = dataSetDF.describe()
print(dataSetDF.columns)
print(descript)
plt.plot(dataSetDF["X"], dataSetDF['Y'], "ro")
#plt.figure(2)
data = {'a': np.sin(np.linspace(0, 2*pi, 100)),
        'colour': np.random.randint(0, 50, 100),
        'size': np.abs(np.random.randn(50))*100}
data['b'] = data['a'] + 100* np.random.randn(100)

plt.scatter('a', 'b', c='colour', s='size', data=data)
plt.show()
#dataSetDF.plot(kind='scatter', x='X', y='Y',title = "hi")
#plt.scatter('X', 'Y', c='blue', s='50', data = dataSetDF)

#attempt 4: https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
dataSetDF = pd.read_csv("dataset.csv", index_col = 0)
X = dataSetDF.iloc[:,:20].values #converts pandas to numpy
y = dataSetDF.iloc[:,20:21].values #converts pandas to numpy
'''
#attempt 5
#coppied directly from website: https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits #had to import this because I couldn't find data sets anywhere
from sklearn.model_selection import train_test_split

dig = load_digits()
onehot_target = pd.get_dummies(dig.target)

x_train,         x_val,       y_train,       y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)
#training x      testing x    training y     testing y
print(type(y_train))
'''
#this function coppied from madden moore's TP
def partitionTrainAndTest(data, trainSplit):
    assert(type(trainSplit) == int)
    numRows = len(data)
    numRowsToChoose = int(numRows * trainSplit / 100)
    testSplit = 100 - trainSplit
    trainData = pd.DataFrame()
    testData = pd.DataFrame()
    #create list of indicies which will be the training data
    #convert to a set for efficiency
    resultList = set(random.sample(list(range(numRows)), numRowsToChoose))
    for i in range(numRows):
        if i in resultList:
            trainData = trainData.append(data.iloc[i])
        else:
            testData = testData.append(data.iloc[i])
    #split the training and testing data so that the expected
    #results are in one dataframe and the variables are in the other
    diagnosisIndex = 4
    colNames = trainData.columns.tolist()
    dataCols = colNames[:4] + colNames[5:]
    diagnosisCols = [colNames[diagnosisIndex]]
    trainDiagnosis = trainData[diagnosisCols]
    trainData = trainData[dataCols]
    testDiagnosis = testData[diagnosisCols]
    testData = testData[dataCols]
    return trainData, testData, trainDiagnosis, testDiagnosis
    '''
#sigmoid function: used to normalize values
def sigmoid(s):
    return 1/(1 + np.exp(-s))

#derivative of the sigmoid function
def sigmoid_derv(s): 
    return s * (1 - s)

#turns a vector(data) into a probability distrubution
def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

class MyNN:
    def __init__(self, x, y):
        self.x = x
        neurons = 128
        self.lr = 0.5
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3) #soft max takes vectors and turns them into probablity distributions
        
    def backprop(self):
        loss = error(self.a3, self.y)
        #print('Error :', loss)
        a3_delta = cross_entropy(self.a3, self.y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    #predicts from the testing data set by going forward in the NN
    #used: to get the accuracy
    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax() #returns the highest of the probability distrubution (most probable value)
			
#object of NN class
model = MyNN(x_train/16.0, np.array(y_train))

epochs = 1500 #full cycles of the NN through the training set
#runs the model forward and back the number of epochs through the training set
for x in range(epochs):
    model.feedforward()
    model.backprop()
		
#returns the accuracy of the model
#if the prediction is the same as the biggest vaule in y, accuracy goes up by 1
#used: to get the accuracy of the model for the training data and the testing data
def get_acc(x, y): 
    acc = 0
    for xx,yy in zip(x, y): #zip function makes tuples from x and y interables
        if model.predict(xx) == np.argmax(yy): #np.argmax returns the indices of the maximum values along an axis. 
            acc +=1
    return acc/len(x)*100 #returns the accuracy as a percentage of the data set accurately predicted
	
print("Training accuracy : ", get_acc(x_train/16, np.array(y_train)))#div by 16 because 16 is the largest value in the set (makes it all between 0 and 1)
print("Test accuracy : ", get_acc(x_val/16, np.array(y_val)))
