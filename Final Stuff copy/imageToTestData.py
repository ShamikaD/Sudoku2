from PIL import Image
import numpy as np
import pandas as pd
import pickle

#this makes the image into a 2d Array of 1s and 0s where 1 is a white space 
# and 0 is a black (written in) space
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

#gets rid of the empty rows above the letter(s)
def getRidOfEmptyStartRows(img):
    i=0
    while i < img.shape[0]:
        if 0 in img[i]:
            return img[i:]
        i+=1
    return np.array([])

#gets rid of the empty cols to the left of the letter(s)
def getRidOfEmptyStartCols(line):
    line = pd.DataFrame(line)
    i = 0
    for (name, col) in line.iteritems():
        col = np.array(col)
        if 0 in col:
            return np.array(line.drop(line.columns[range(i)],axis=1))
        i+=1
    return np.array([])

#resizes the array from user to match the dimentions of the training data 
# used for the neural network
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
    #making the cols 128
    addToLeft = (128 - letter.shape[1])//2
    left = np.ones((addToLeft, letter.shape[0]), dtype=int)
    letter = np.concatenate((left.T, letter), axis=1)
    addToRight = 128 - letter.shape[1]
    right = np.ones((addToRight, letter.shape[0]), dtype=int)
    letter = np.concatenate((letter, right.T), axis=1)
    #making the rows 128
    addToTop = (128 - letter.shape[0])//2
    top = np.ones((addToTop, letter.shape[1]), dtype=int)
    letter = np.concatenate((top, letter), axis=0)
    addToBottom = 128 - letter.shape[0]
    bottom = np.ones((addToBottom, letter.shape[1]), dtype=int)
    letter = np.concatenate((letter, bottom), axis=0)
    return letter

#returns an array of the arrays of indivisual lines when given the array of a 
# whole image of text
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
    return np.array(lines)
    
#returns a list of the arrays of indivisual letter when given the array of an 
# image of a line of letters
def getLettersFromLine(line, letters):
    if len(line) == 0:
        return np.array(letters)
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

#flattens the 2D array to one long np array
def flatten(letter):
    let = np.array([])
    for i in range(128):
        let = np.concatenate((let,np.array(letter[i])), axis = None)
    return (np.array(let.astype(float)))

#returns the letter data from an image in a 3d array (one line per row and one array per letter in the row)
# in a format that is good for the neural network to be able to predict from
def getAllLetters(imageName):
    pic = np.array(Image.open(imageName), np.int)
    binPic = makeBinary(pic)
    lines = getLines(binPic, [])
    allLetters = []
    for line in lines:
        allLetters.append(getLettersFromLine(line, []))
        #all Letters becomes a list of arrays of each letter
        #they are all 128 by 128
    lets = []
    for line in allLetters:
        newLin = []
        for i in range(line.shape[0]):
            newLin.append(flatten(line[i]))
        lets.append(np.array(newLin))
    return np.array(lets)

real = 'abc.png'
#print("This code complies! :)")
#getLetters(real)
print(getAllLetters(real))
