from PIL import Image
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

#this makes the image into a 2d Array of 1s and 0s where 1 is a white space 
# and 0 is a black (written in) space
def makeBinary(pic):
    binPic = []
    for row in pic:
        ncol = []
        for col in row:
            ncol.append((col[0]+col[1])/(2*255))
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
    '''
    resizeTo = 6
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
    '''
     #this is adding back the white space around each letter to make it an even square
    if letter.shape[0] > letter.shape[1]:
        addToLeft = (letter.shape[0] - letter.shape[1])//2
        left = np.ones((addToLeft, letter.shape[0]), dtype=int)
        letter = np.concatenate((left.T, letter), axis=1)
        addToRight = letter.shape[0] - letter.shape[1]
        right = np.ones((addToRight, letter.shape[0]), dtype=int)
        letter = np.concatenate((letter, right.T), axis=1)
    elif letter.shape[0] < letter.shape[1]:
        addToTop = (letter.shape[1] - letter.shape[0])//2
        top = np.ones((addToTop, letter.shape[1]), dtype=int)
        letter = np.concatenate((top, letter), axis=0)
        addToBottom = letter.shape[1] - letter.shape[0]
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

#returns the letter data from an image in a 3d array (one line per row and one array per letter in the row)
# in a format that is good for the neural network to be able to predict from
def getAllLetters(imageName):
    pic = np.array(Image.open('123.png').convert('LA'), np.float)
    grey = makeBinary(pic)
    lines = getLines(grey, [])
    allLetters = []
    for line in lines:
        allLetters.append(getLettersFromLine(line, []))
        #all Letters becomes a list of arrays of each letter
        #they are all 128 by 128
    lets = []
    for line in allLetters:
        newLin = []
        for i in range(line.shape[0]):
            newArry = line[i].flatten()
            newLin.append(newArry)
        lets.append(np.array(newLin))
    return np.array(lets)
'''
real = '123.png'
#print("This code complies! :)")
#getLetters(real)
#pic = np.array(Image.open('123.png').convert('LA'), np.float)
#grey = makeBinary(pic)
#print(grey)
#print(getAllLetters(real)[0][0].shape)
#with open('numbers.pickle', 'wb') as handle:
 #   pickle.dump(getAllLetters(real), handle, protocol=pickle.HIGHEST_PROTOCOL)

#first number in dig.data (it's a 0)

[ 0.  0.  5. 13.  9.  1.  0.  0.  
  0.  0. 13. 15. 10. 15.  5.  0.  
  0.  3. 15.  2.  0. 11.  8.  0.  
  0.  4. 12.  0.  0.  8.  8.  0.  
  0.  5.  8.  0.  0.  9.  8.  0.  
  0.  4. 11.  0.  1. 12.  7.  0.  
  0.  2. 14.  5. 10. 12.  0.  0.  
  0.  0.  6. 13. 10.  0.  0.  0.]

#https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil 
img = Image.open(real)
img.load() # required for png.split()

imgRGB = Image.new("RGB", img.size, (255, 255, 255))
imgRGB.paste(img, mask=img.split()[3]) # 3 is the alpha channel
rgbName = real[:-4]+'RGB.png'
#https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
imgRGB.save(rgbName, format = None, quality=80)

#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imread.html
img = mpl.image.imread(rgbName)
#https://kite.com/python/answers/how-to-convert-an-image-from-rgb-to-grayscale-in-python
rgbWeights = [0.2989, 0.5870, 0.1140]
grayscale_image = np.dot(img[...,:3], rgbWeights)
print(grayscale_image.shape)
#plt.imshow(grayscale_image, cmap=plt.get_cmap("gray"))
#plt.imshow(grayscale_image, cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()
'''