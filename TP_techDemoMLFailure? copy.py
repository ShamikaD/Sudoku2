import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#from tutorial: https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/

#basically, we will asign two vectors, bias and weight to each input on each layer
#then we will use sigmoid function (good because between 1 and 0) to "vote" true or False
#then these votes will go to the next layer and get tallied to do the same thing
#eventually, you get an output that is either a 1 or a 0

#sets the parameters for each layer
def makeParams (layerDimensions):
    seed = np.random.seed(3) #sets the seed for a pseudo random number generator
    params = {} #dictionary of parameters for the layer
    numDims = len(layerDimensions)
    for i in range (1, numDims):
        currWeight = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        params["Weight" + str(i)] = currWeight
        params["Bias" + str(i)] = np.zeros((layer_dims[i], 1)) #makes list of 0s the length of i
    return params

# Z (linear hypothesis) - Z = W*X + b , 
#activation function: makes linear outputs to non-linear outputs (turns into 1 or 0) 
def sigmoid(linHyp):
    dotProd = np.dot(-1, linHyp)
    result = 1/(1+np.exp(dotProd))
    cache = (linHyp) #stores linHyp for use in backpropogation
    return result, cache
 
#takes the values from the previous layer and gives them to the next layer
def passOn(trainingData, params):
    #skipped line A = X because it seemed unneeded
    caches = [] #to save data in later
    for i in range(1, (len(params) // 2) + 1):
        lastData = trainingData #loops through every layer 
        linHyp = np.dot(params['Weight'+str(i)], lastData) + params['Bias'+str(i)] 
        linCache = (lastData, params['Weight'+str(i)], params['Bias'+str(i)]) 
        trainingData, actCache = sigmoid(linHyp) 
        caches.append((linCache, actCache)) #stores both types of caches as a tuple
    return trainingData, caches

#calculates the cost of our NN. The lower the cost, the bette the performance
#we want to minimize the cost function so that out NN is better to run
def cost(trainingData, Y): 
    '''WHAT IS Y!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    return ((-1/Y.shape[1])*(np.dot(np.log(trainingData), Y.T))+np.dot(log(1-trainingData), 1-Y.T))

#One way to reduce the cost function is by using gradient descent: 
#calculates change in the final output with repect to the change in parameters of a 
#particular neuron. Used in backpropogation
def oneLayerBack(changeTraining, cache):
    linCache, actCache = cache
    changeAct = changeTraining*sigmoid(actCache)*(1-sigmoid(actCache)) #derivative of sigmoid
    lastData, weight, bias = linCache
    changeWeight = (1/lastData.shape[1])*np.dot(changeAct, lastData.T)
    changeBias = (1/lastData.shape[1])*np.sum(changeAct, axis=1, keepdims=True)
    changeLastData = np.dot(weight.T, changeAct)
    return changeLastData, changeWeight, changeBias #derivatives of cost function

#computes and returns all the gradients for all the layers
def backPropogation(trainingLayer, Y, caches):
    gradients = {}
    amountCaches = len(caches)
    Y = Y.reshape(trainingLayer.shape)
    changeTrainingLayer = -(np.divide(Y,trainingLayer)-np.divide(1-Y,1-trianingLayer))
    currCache = caches[amountCaches-1]
    gradients['change training'+str(amountCaches-1)], \
        gradients['change weight'+str(amountCaches-1)], \
            gradients['change bias'+str(amountCaches-1)] \
                = oneLayerBack(changeTrainingLayer, currCache)
    for i in range(amountCaches-1, 0, -1):
        currCache = caches[1]
        changeLastTrainTemp, changeWeightTemp, changeBiasTemp = oneLayerBack(
                gradients["change training" + str(i+1)], currCache)
        gradients["change training" + str(i)] = changeLastTrainTemp
        gradients["change weight" + str(i + 1)] = changeWeightTemp
        gradients["change bias" + str(i + 1)] = changeBiasTemp
    return gradients

#updates the parameters in all the layers
def updateParams(parameters, gradients, learningRate):
    for i in range(len(parameters) // 2):
        parameters['Weight' + str(i+1)] = parameters['Weight'+str(i+1)] -\
            learningRate * gradients['Weight'+str(i+1)]
        parameters['Bias' + str(i+1)] = parameters['Bias'+str(i+1)] -\
            learningRate * gradients['Bias'+str(i+1)]
    return parameters

def train(trainingData, Y, layerDimentions, epochs, learningRate):
    params = makeParams(layerDimentions)
    costHistory = [] #tells you how good your network architecture is
    for i in range(epochs):
        Y_hat, caches = passOn(trainingData, params)
        cost = cost(Y_hat, Y)
        costHistory.append(cost)
        gradients = backprop(Y_hat, Y, caches)
        
        params = updateParams(params, gradients, learningRate)
    return params, costHistory

print("\nCompiled! :)")

    