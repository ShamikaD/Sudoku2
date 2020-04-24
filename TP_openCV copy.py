from PIL import Image
import numpy as np
import pandas as pd
import pickle

############################################################################################
# plan:
#   √  get rid of all starting rows with only 1s
#   √  go through every row and look for a row with all 1s (after clearing all starting rows)
#   √  split all prev rows (with 0s) into a different array
#   √  keep repeating these steps until you isolate every line
#
#   √  get rid of all starting cols with only 1s
#   √  go through every col and look for a col with all 1s (after clearing all starting cols)
#   √  split all prev cols (with 0s) into a different array
#   √  keep repeating these steps until you isolate every letter
#   resize and fix letters to match with training set of neural net
#   feed the letters (in order into the neural net)
# 
# To figure out later: how to make the letters straight
#                      how to clean up the image (no stray/grey marks and make greyscale)
#                        --> This is only really needed for people who just want a cleaner image returned
############################################################################################  
#tutorial for image processing here (tells how to open images with PIL): https://note.nkmk.me/en/python-numpy-image-processing/

'''
#gets rid of transparency values (takes a really long time and is (maybe0 not really needed)
nonTransparent = np.array([])
count = 0
if (len(pic[0][0])) == 4:
    for row in pic:
        print("working... ",920-count)
        newRow = np.array([])
        for col in pic:
            newCol = pd.DataFrame(col)
            newCol = np.array(newCol.drop(newCol.columns[-1],axis=1))
            #for rgb in col:
                #np.append(newCol, rgb[:-1])
            np.append(newRow, newCol)
        np.append(nonTransparent, newRow)
        count+=1
pic = nonTransparent

#to clean up:
#keep going through lists until you hit a list made of ints
#then go through every int and if the avg of the ints in that list is not less than 130(about 255/2), 
# make all the elem in the list = 255 otherwise, make all elem 0

#cleanedPic = Image.fromarray(pic.astype(np.uint8))
#cleanedPic.save('abc2.png')
'''
#this makes the image into a 2d Array of 1s and 0s where 1 is a white space and 0 is a black space
#maybe to clean up the image, convert to binary like this and then convert back to rgb and make png
def makeBinary(pic):
    binPic = []
    for row in pic:
        ncol = []
        for col in row:
            if col[0] >= 100:
                ncol.append(1)
            else:
                ncol.append(0)
        binPic.append(ncol)
    return np.array(binPic)

def getRidOfEmptyStartRows(img):
    i=0
    while i < img.shape[0]:
        if 0 in img[i]:
            return img[i:]
        i+=1
    return np.array([])

#makes the height of the letter similar to the height of the training data
def resize (letter):
    resizeTo = 30
    #this is resizing the height
    toSkip = letter.shape[0]//resizeTo
    corrHeightPic = []
    i=0
    while i < letter.shape[0]:
        corrHeightPic.append(letter[i])
        i+= toSkip
    letter = np.array(corrHeightPic)
    #this is resizing the width
    i=0
    toDrop = []
    while i < letter.shape[1]:
        if i%toSkip!=0:
            toDrop.append(i)
        i+= 1
    letter = pd.DataFrame(letter)
    letter = np.array(letter.drop(letter.columns[toDrop],axis=1))
    #this is adding back the white space around each letter to make it 128 by 128
    addToLeft = (128 - letter.shape[1])//2
    left = np.ones((addToLeft, letter.shape[0]), dtype=int)
    letter = np.concatenate((left.T, letter), axis=1)
    addToRight = 128 - letter.shape[1]
    right = np.ones((addToRight, letter.shape[0]), dtype=int)
    letter = np.concatenate((letter, right.T), axis=1)

    addToTop = (128 - letter.shape[0])//2
    top = np.ones((addToTop, letter.shape[1]), dtype=int)
    letter = np.concatenate((top, letter), axis=0)
    addToBottom = 128 - letter.shape[0]
    bottom = np.ones((addToBottom, letter.shape[1]), dtype=int)
    letter = np.concatenate((letter, bottom), axis=0)
    return letter

def getRidOfEmptyStartCols(line):
    line = pd.DataFrame(line)
    i = 0
    for (name, col) in line.iteritems():
        col = np.array(col)
        if 0 in col:
            return np.array(line.drop(line.columns[range(i)],axis=1))
        i+=1
    return np.array([])

def getLettersFromLine(line, letters):
    if len(line) == 0:
        return letters 
    else:
        line = pd.DataFrame(getRidOfEmptyStartCols(line))
        i = 0
        for (name, col) in line.iteritems():
            col = np.array(col)
            if 0 not in col:
                letterCols = list(range(i))
                nonLetterCols = list(range(i, len(line.columns)))
                letter = np.array(line.drop(line.columns[nonLetterCols],axis=1))
                if letter.shape[1] >= 20:
                    letters.append(resize(letter))
                newLine = line.drop(line.columns[letterCols],axis=1)
                return getLettersFromLine(np.array(newLine), letters)
            i+=1

def getLines(image, lines):
    if image.size != 0:
        image = getRidOfEmptyStartRows(image)
        i = 0
        for row in image:
            if 0 not in row:
                line = image[:i]
                if line.shape[0] >= 20:
                    lines.append(line)
                restOfLines = image[i:]
                return getLines(restOfLines, lines)
            i+=1
    return lines

def flatten(arr):
    newArr = []
    for l in arr: #this is to flatten the array
        newArr.extend(l)
    return pd.DataFrame(newArr)

def getAllLetters(imageName):
    pic = np.array(Image.open(imageName), np.int)
    binPic = makeBinary(pic)
    lines = getLines(binPic, [])
    allLetters = []
    for line in lines:
        allLetters.append(getLettersFromLine(line, []))
    for l in allLetters:
        for letters in l:
            print(letters.shape)
            np.savetxt('test.csv', letters, delimiter=',')
    return allLetters

real = 'helloBob.png'
print("This code complies! :)")
getAllLetters(real)
