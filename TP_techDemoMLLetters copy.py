#This file is just work making the data set workable

from PIL import Image
import numpy as np
import pickle
import os
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
#storing stuff as pickle 
with open('allLetterData.pickle', 'wb') as handle:
    pickle.dump(allData, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
#when I want to open my pickle file later
with open('allLetterData.pickle', 'rb') as handle:
    openedFile = pickle.load(handle)
'''

#pickling: https://www.datacamp.com/community/tutorials/pickle-python-tutorial
'''
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
'''
print("complies :)")