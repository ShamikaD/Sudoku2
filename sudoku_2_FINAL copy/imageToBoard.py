#This file converts the images into a 2D list of numbers that the neural network can read from

#pandas, numpy, and matplotlib all came with anacoda, which I downloaded
# from here: https://docs.anaconda.com/anaconda/install/

#I taught myself pandas from: 
#https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/
#The rest of the pandas information that I learned is from: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
#I taught myself matplotlib from: https://matplotlib.org/users/pyplot_tutorial.html
#the rest of the matplotlib information is from: https://matplotlib.org/tutorials/index.html#introductory
#I taught myself numpy from: https://docs.scipy.org/doc/numpy/user/quickstart.html
#the rest of the numpy information is from: https://docs.scipy.org/doc/numpy/reference/

#PIL installation instructions from here: http://www.cs.cmu.edu/~112/notes/notes-animations-part2.html#spritesheetsWithCropping
#I taught myself basic PIL from here: https://pillow.readthedocs.io/en/stable/handbook/tutorial.html
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def makeGrey(fileName):
    #Making images RGB from RGBA: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil 
    img = Image.open(fileName)
    img.load() # required for img.split()
    imgRGB = Image.new("RGB", img.size, (255, 255, 255))
    imgRGB.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    rgbName = fileName[:-4]+'RGB.png'
    #Saving Images: https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
    imgRGB.save(rgbName, format = None, quality=80)
    #Reading Images with matplotlib: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imread.html
    img = mpl.image.imread(rgbName)
    os.remove(rgbName)
    #https://kite.com/python/answers/how-to-convert-an-image-from-rgb-to-grayscale-in-python
    rgbWeights = [0.2989, 0.5870, 0.1140]
    grey = np.dot(img[...,:3], rgbWeights)
    return grey

#this makes the image into a 2d Array of 1s and 0s where 1 is a white space 
# and 0 is a black (written in) space
def makeBinary(fileName):
    pic = np.array(Image.open(fileName), np.int)
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
def getRidOfEmptyStartRows(binPic, greyPic):
    i=0
    while i < binPic.shape[0]:
        if 0 in binPic[i]:
            return binPic[i:], greyPic[i:]
        i+=1
    return np.array([]), np.array([])

#gets rid of the empty cols to the left of the letter(s)
def getRidOfEmptyStartCols(binLine, greyLine):
    temp = binLine
    binLine = pd.DataFrame(binLine)
    greyLine = pd.DataFrame(greyLine)
    i = 0
    for j in range(temp.shape[1]):
        binCol = temp[:,j]
        if 0 in binCol:
            binRest = np.array(binLine.drop(binLine.columns[range(i)],axis=1))
            greyRest = np.array(greyLine.drop(greyLine.columns[range(i)],axis=1))
            return binRest, greyRest
        i+=1
    return np.array([]), np.array([])

#resizes the array from user to match the dimentions of the training data 
# used for the neural network
def resize (letter, isBin):
     #this is adding back the white space around each letter to make it an even square
    if letter.shape[0] > letter.shape[1]:
        addToLeft = (letter.shape[0] - letter.shape[1])//2
        if (isBin):
            left = np.ones((addToLeft, letter.shape[0]), dtype=int)
        else:
            left = np.full((addToLeft, letter.shape[0]), 0.9998999999999999, dtype=float)
            #I chose this number to fill the array with because it was the closest to 1 that was not 1
        letter = np.concatenate((left.T, letter), axis=1)
        addToRight = letter.shape[0] - letter.shape[1]
        if (isBin):
            right = np.ones((addToRight, letter.shape[0]), dtype=int)
        else:
            right = np.full((addToRight, letter.shape[0]), 0.9998999999999999, dtype=float)
        letter = np.concatenate((letter, right.T), axis=1)
    elif letter.shape[0] < letter.shape[1]:
        addToTop = (letter.shape[1] - letter.shape[0])//2
        if (isBin):
            top = np.ones((addToTop, letter.shape[1]), dtype=int)
        else:
            top = np.full((addToTop, letter.shape[1]), 0.9998999999999999, dtype=float)
        letter = np.concatenate((top, letter), axis=0)
        addToBottom = letter.shape[1] - letter.shape[0]
        if (isBin):
            bottom = np.ones((addToBottom, letter.shape[1]), dtype=int)
        else:
            bottom = np.full((addToBottom, letter.shape[1]), 0.9998999999999999, dtype=float)
        letter = np.concatenate((letter, bottom), axis=0)
    return letter

#returns an array of the arrays of indivisual lines when given the array of a 
# whole image of text
def getLines(binPic, binLines, greyPic, greyLines):
    if binPic.size != 0:
        binPic, greyPic = getRidOfEmptyStartRows(binPic, greyPic)
        i = 0
        for row in binPic:
            if 0 not in row:
                binLine = binPic[:i]
                greyLine = greyPic[:i]
                if binLine.shape[0] >= 20:
                    binLines.append(binLine)
                    greyLines.append(greyLine)
                restOfBinLines = binPic[i:]
                restOfGreyLines = greyPic[i:]
                return getLines(restOfBinLines, binLines, restOfGreyLines, greyLines)
            i+=1
    return np.array(binLines), np.array(greyLines)

#returns a list of the arrays of indivisual letter when given the array of an 
# image of a line of letters
def getLettersFromLine(binLine, binLetters, greyLine, greyLetters):
    if len(binLine) == 0:
        return np.array(binLetters), np.array(greyLetters)
    else:
        binLine, greyLine = getRidOfEmptyStartCols(binLine, greyLine)
        temp = binLine
        binLine = pd.DataFrame(binLine)
        greyLine = pd.DataFrame(greyLine)
        i = 0
        for (name, col) in binLine.iteritems():
            col = np.array(col)
            if 0 not in col:
                binLetterCols = list(range(i))
                greyLetterCols = list(range(i))
                nonBinLetterCols = list(range(i, len(binLine.columns)))
                nonGreyLetterCols = list(range(i, len(greyLine.columns)))
                binLetter = np.array(binLine.drop(binLine.columns[nonBinLetterCols],axis=1))
                greyLetter = np.array(greyLine.drop(greyLine.columns[nonGreyLetterCols],axis=1))
                if binLetter.shape[1] >= 20:
                    binLetters.append(resize(binLetter, True))
                    greyLetters.append(resize(greyLetter, False))
                newBinLine = binLine.drop(binLine.columns[binLetterCols],axis=1)
                newGreyLine = greyLine.drop(greyLine.columns[greyLetterCols],axis=1)
                return getLettersFromLine(np.array(newBinLine), binLetters, np.array(newGreyLine), greyLetters)
            i+=1

#returns the letter data from an image in a 3d array (one line per row and one array per letter in the row)
# in a format that is good for the neural network to be able to predict from
def getAllLetters(fileName):
    binPic = makeBinary(fileName)
    greyPic = makeGrey(fileName)
    binLines, greyLines = getLines(binPic, [], greyPic, [])
    allBinLetters = [] 
    allGreyLetters = []
    for i in range(len(binLines)):
        binLet, greyLet = getLettersFromLine(binLines[i], [], greyLines[i], [])
        allBinLetters.append(binLet)
        allGreyLetters.append(greyLet)
    binLets = []
    greyLets = []
    for j in range(len(allBinLetters)):
        newBinLin = []
        newGreyLin = []
        for i in range(allBinLetters[j].shape[0]):
            newBinLin.append(allBinLetters[j][i].flatten())
            newGreyLin.append(allGreyLetters[j][i])
        binLets.append(np.array(newBinLin))
        greyLets.append(newGreyLin)
    return greyLets

#makes all the images into a format that is close to the training data for the neural network
def forNet(fileName):
    lets = getAllLetters(fileName)
    lines = []
    for i in range(len(lets)):
        letters = []
        for j in range(len(lets[i])):
            img = lets[i][j]
            #rescaling greyscale to be able to become a png:
            # https://stackoverflow.com/questions/6915106/saving-a-numpy-array-as-an-image-instructions
            grey = ((255.0/img.max())*(img-img.min())).astype(np.uint8)
            im = Image.fromarray(grey)
            newFileName = fileName[:-4]+ str((len(lets[i])*i) + j) + '.png'
            im.save(newFileName)
            img = Image.open(newFileName)
            os.remove(newFileName)
            #resizing the image to 8 by 8 (the size of the training data)
            # https://stackoverflow.com/questions/47143332/how-to-pixelate-a-square-image-to-256-big-pixels-with-python
            imgPix = img.resize((8,8),resample = Image.BILINEAR)
            pixFile = newFileName[:-4] + 'result.png'
            imgPix.save(pixFile)
            img = mpl.image.imread(pixFile).flatten()
            os.remove(pixFile)
            #scaling the image to be the same as the training data (to 16)
            sub = np.ones(img.shape)
            newImg = sub-img
            max = np.amax(newImg)
            scale = 16/max
            letters.append(scale*newImg)
        lines.append(letters)
    return lines