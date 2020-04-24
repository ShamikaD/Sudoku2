#This file is just work making the data set workable

from PIL import Image
import numpy as np
import pandas as pd
import random #needed for random partitioning
import pickle #needed for saving files
import os #needed for traversing through files
import time # needed to see how long things take
start_time = time.time()

'''
#tutorial for image processing here: https://note.nkmk.me/en/python-numpy-image-processing/
#to make an image into an array
pic = np.array(Image.open('z.png'), np.float) #makes a 3d array of floats
#all my images are size 128 by 128

#to save an array image as a file
newImage = Image.fromarray(pic.astype(np.uint8))
#newImage.save('NEW_NAME.png')

#reading through file paths: https://www.guru99.com/reading-and-writing-files-in-python.html
#opening a txt file
file = open("FILENAME.txt","w+")
    #w = write, r = read, a = append, + = read and write
    #if there is a +, it also will make you a file of that name to write on if 
          #there is no preexisting file with that name

#some sample file writing
for i in range(10):
     file.write("This is line %d\r\n" % (i+1))
     #if you do file.write the first time after you opened the file, it deletes 
     #whatever was already on the file ans starts fresh

file.close() #closes the file
file = open("FILENAME.txt","w+")

#############################################################################
#                      Code Works: takes forever to run                     #
#############################################################################

#file paths tutorial: https://realpython.com/working-with-files-in-python/#pythons-with-open-as-pattern
allFolders = os.listdir('lettersData')
y = np.array([])
x = np.array([])
count = 1
for letter in allFolders:
    if letter != '.DS_Store': #this is a mac storage thing that I dont want to open but can't get rid of
        letterFiles = os.listdir('lettersData/' + letter)
       #y = np.append(y, letter) #do this for every x that you append
        for folder in letterFiles:
            if folder != '.DS_Store' and 'mit' not in folder:
                images = os.listdir('lettersData/' + letter + "/" + folder)
                for image in images:
                    x = np.append(x, np.array(Image.open('lettersData/' + letter 
                        + "/" + folder + "/" + image), np.float))
                    y = np.append(y, letter)
                    count += 1
                    print("running... " + str(count))

#############################################################################
#from: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html

x = np.array([1,2,1,1])
r = np.array([[[1],[2]],[1],[1]])
#assert(x==r)

np.savetxt("x.csv", x) #this won't actually work because it is looking for a 1d or 2d array
                       #so, you havet to rearrance the array in np first
np.savetxt("y.csv", y)

from numpy import genfromtxt
my_data = genfromtxt('hi.csv', delimiter=',')
print(my_data)

'''
#file paths tutorial: https://realpython.com/working-with-files-in-python/#pythons-with-open-as-pattern
#this code makes all the images from the lettersData folder into binary 2D arrays
# with 1 as a white space and 0 as a black space
#it then saves all of these 2D arrays in a dictionary that maps each letter to all 
# of the images of that letter
count = 1
allData = {}
allFolders = os.listdir('lettersData')
for letter in allFolders:
    if letter != '.DS_Store': #this is a mac storage thing that I dont want to open but can't get rid of
        allData[letter] = []
        letterFiles = os.listdir('lettersData/' + letter)
        imageCount = 0
        for folder in letterFiles:
            if imageCount < 6000 and folder != '.DS_Store' and 'mit' not in folder:
                images = os.listdir('lettersData/' + letter + "/" + folder)
                for image in images:
                    newIM = []
                    img = np.array(Image.open('lettersData/' + letter 
                        + "/" + folder + "/" + image), np.int)
                    for row in img:
                        ncol = []
                        for col in row:
                            if col[0] >= 100:
                                ncol.append(1)
                            else:
                                ncol.append(0)
                        newIM.append(ncol)
                    allData[letter].append(newIM)
                    imageCount+=1
                    count+=1
                    print("running... " +letter+ str(count))
print("done :)")
#storing all the data as a pickle file so I never have to run it again
with open('allLetterData.pickle', 'wb') as handle:
    pickle.dump(allData, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''

#pickling: https://www.datacamp.com/community/tutorials/pickle-python-tutorial
#This is code to make sure my converting to binary code works
igm = np.array(Image.open("lettersData/a/hsf_0/hsf_0_00000.png"), np.int)
newIM = []
for row in igm:
    ncol = []
    for col in row:
        if col[0] >= 100:
            ncol.append(1)
        else:
            ncol.append(0)
    newIM.append(ncol)
arr = np.array(newIM)
print(str(arr))
print(arr.shape)
b= np.savetxt("EXPERIMENT2.csv", arr)
print (b)

#this code opens the saved dictionary of letter data and converts it to a data frame 
# where the letter is the index (vertical label for each row) and each row is the corresponding array dor wach image
#this code takes 22 mins to run (ll for opening the pickle and 11 for converting)
with open('allLetterData.pickle', 'rb') as handle:
    openedFile = pickle.load(handle)
#https://pandas.pydata.org/pandas-docs/version/0.23.1/generated/pandas.DataFrame.from_dict.html
data = pd.DataFrame.from_dict(openedFile, orient='index')

#This code separates all the X (arrays of binary) from the Y (actual letters) in 
# the data and cleans up the arrays in each col by flattening them
    #The flattening here didn't really work and made the arrays into arrays of 128 lists of 128 elements 
    # instead of making one long array of 16384 elements for each col in the dataframe
#it also converts the letters to strings of binary so that it can be read more 
# easily by the neural net
#this code took 11 hours to complete

#converts letters into binary
def letterConvert(letter):
    zeroes = [0] * 26
    num = ord(letter) - ord("a")
    zeroes[num] = 1
    return pd.Series(zeroes)

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

#This code partitions all the data so that there is a 80-20 split (randomly) of training and testing data
#partitioning took around 9 hours to complete
with open('allX.pickle', 'rb') as handle:
    x = pickle.load(handle)
with open('allY.pickle', 'rb') as handle:
    y = pickle.load(handle)

#partitions the data into trainin and testing, and input and output
#Partition algorithm based off of (kind of): https://github.com/maddenmoore/neural-network-visualizer 
def partition(x, y):
    numDataPts = len(x.index)
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

trainX, testX, trainY, testY = partition(x, y)

#This code is to fix the mistake that I made earlier when trying to flatten the array
#it saves the training and testing X data as a 2D numpy array that has rows for each letter and each row is 
# 16384 digits of binary for that image
3this code takes about 35 mins to run
def flatten(letter):
    let = np.array([])
    for i in range(128):
        let = np.concatenate((let,np.array(letter[i])), axis = None)
    return (np.array(let.astype(float)))

with open('trainX.pickle', 'rb') as handle:
    trainX = pickle.load(handle)

with open('testX.pickle', 'rb') as handle:
    testX = pickle.load(handle)

trainX2 = []
for i in range(trainX.shape[0]):
    trainX2.append(flatten(trainX[i]))
    print ("Train: Fixing ", i, " letters took ", (time.time() - start_time)/60, " mins to run")

testX2 = []
for i in range(testX.shape[0]):
    testX2.append(flatten(testX[i]))
    print ("Train: Fixing ", i, " letters took ", (time.time() - start_time)/60, " mins to run")

with open('trainX3.pickle', 'wb') as handle:
    pickle.dump(np.array(trainX2), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('testX3.pickle', 'wb') as handle:
    pickle.dump(np.array(testX2), handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
print("complies :)")

