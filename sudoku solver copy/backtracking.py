#numpy came with anacoda, which I downloaded from: https://docs.anaconda.com/anaconda/install/
import numpy as np
import math, copy
#Checking if the board is legal my work from 15-112 homework 5 from this year
#guidelines for that homework are here: http://www.cs.cmu.edu/~112/notes/hw5.html

#returns if a given sudoku board is legal (it has all allowed numbers in its 
# rows, columns, and boxes)
def isLegalSudoku(board):
    for i in range(len(board)):
        if (not isLegalRow(board, i) or not isLegalCol(board, i) 
                or not isLegalBlock(board, i)):
            return False
    return True

#checks if the values in a given list are legal for a game of sudoku
def areLegalValues(values):
    check = []
    blockLength = math.sqrt(len(values))
    if blockLength != int(blockLength):
        return False
    legal = list(range(1, int(blockLength) ** 2 + 1))
    for num in values:
        if num != 0 and (num in check or not num in legal):
            return False
        check += [num]
    return True

#Returns if a specified row is legal on a sudoku board
def isLegalRow(board, row):
    return areLegalValues(board[row])

#Returns if a specified column is legal on a sudoku board
def isLegalCol(board, col):
    column = []
    for row in board:
        column.append(row[col])
    return areLegalValues(column)

#Returns if a specified block is legal on a sudoku board
def isLegalBlock(board, block):
    blockLength = int(math.sqrt(len(board)))
    #The following makes a list of integers that represent where the block 
    # would be if each block is one element in a 2D list
    findBlock = []
    count = 0
    for row in range(blockLength):
        newCol = []
        for col in range(blockLength):
            newCol.append(count)
            count += 1
        findBlock.append(newCol)
    blockCoords = findCoords(findBlock, block)
    #the following makes a list of the ints that are in that are in a block
    startCoord = [blockLength * blockCoords[0], blockLength * blockCoords[1]]
    resultBlock = []
    for row in range(blockLength):
        for col in range(blockLength):
            resultBlock += [board[startCoord[0] + row][startCoord[1] + col]]
    return areLegalValues(resultBlock)

#Returns where on the board the current block is 
def findCoords(board, target):
    for row in range (len(board)):
        newCol = []
        for col in range (len(board[0])):
            if board[row][col] == target:
                return (row,col)
         
#returns the solved puzzle (wrapper function for solver)
def solvePuzzle(board):
    numbers = [1,2,3,4,5,6,7,8,9]
    poss = getPossibleSpots(board)
    return solver(numbers, board, poss)
    
#returns the solved puzzle
def solver(numbers, board, poss):
    if poss==[]:
        return board
    else:
        spot= poss[0]
        #tries to place every number into each spot
        for number in numbers:
            #makes a copy of the board to check if the move will be legal
            b = copy.deepcopy(board)
            b[spot[0]][spot[1]] = number
            if isLegalSudoku(b) and board[spot[0]][spot[1]] == 0:
                board[spot[0]][spot[1]] = number
                #recursive call to see if it needs to backtrack anywhere
                solved = solver(numbers, board, poss[1:])
                if solved != False:
                    return solved
                else:
                    #resets the space for backtracking
                    board[spot[0]][spot[1]] = 0
        return False

#returns a list of indicies where there are open spots
def getPossibleSpots(board):
    spots = []
    for i in range (len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                spots.append((i,j))
    return spots

# returns a list of all the values in the given list without any nested lists
def flatten(tot):#nested, first = True): 
    bob = []
    for l in tot:
        bob.extend(l)
    return bob

