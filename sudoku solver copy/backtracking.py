import numpy as np
import math, copy
#This is my work from homework 5 
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


    

    ##########################################################################
    #           check if the board is legal                                  #
    #           if it is, solve using backtracking                           #
    ##########################################################################      
    
def solvePuzzle(board):
    numbers = [1,2,3,4,5,6,7,8,9]
    poss = getPossibleSpots(board)
    print (solver(numbers, board, poss))
    
def solver(numbers, board, poss):
    print2dList (board)
    if 0 not in np.array(board).flatten():
        return board
    else:
        for spot in poss:
            for number in numbers:
                b = copy.deepcopy(board)
                b[spot[0]][spot[1]] = number
                if isLegalSudoku(b):
                    board[spot[0]][spot[1]] = number
                    if (solver(numbers, board, poss)):
                        return True
                    else:
                        board[spot[0]][spot[1]] == 0
                    return False
        return True
'''
                r,c  = possibleSpots[i]
                sol[r][c] = number
                solved = solver(possibleSpots[1:], numbers, sol)
                if solved != None:
                    return solved
                else:
                    sol[r][c] = 0
        return None



    def solver(rules, aPosition, letters, sol):
        if 0 not in flatten(sol):
            return sol
        else:
            #tries to place a number in the next legal position 
            number = numbers[0]
            directions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),  (0, 0),  (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
            for change in directions:
                row = aPosition[0] + change[0]
                col = aPosition[1] + change[1]
                if isLegalSudoku(board) and sol[row][col] == None:
                    sol[row][col] = letter
                    solution = solver(rules, (row, col), letters[1:], sol)
                    if solution != None:
                        return solution
                    else:
                        sol[row][col] = None
            return None
'''
def getPossibleSpots(board):
    spots = []
    for i in range (len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                spots.append((i,j))
    return spots

board = [
    [ 0, 0, 0, 1, 0, 5, 0, 0, 0 ],
    [ 1, 4, 0, 0, 0, 0, 6, 7, 0 ],
    [ 0, 8, 0, 0, 0, 2, 4, 0, 0 ],
    [ 0, 6, 3, 0, 7, 0, 0, 1, 0 ],
    [ 9, 0, 0, 0, 0, 0, 0, 0, 3 ],
    [ 0, 1, 0, 0, 9, 0, 5, 2, 0 ],
    [ 0, 0, 7, 2, 0, 0, 0, 8, 0 ],
    [ 0, 2, 6, 0, 0, 0, 0, 3, 5 ],
    [ 0, 0, 0, 4, 0, 9, 0, 0, 0 ]
    ]


def maxItemLength(a):
    maxLen = 0
    rows = len(a)
    cols = len(a[0])
    for row in range(rows):
        for col in range(cols):
            maxLen = max(maxLen, len(str(a[row][col])))
    return maxLen

# Because Python prints 2d lists on one row,
# we might want to write our own function
# that prints 2d lists a bit nicer.
def print2dList(a):
    if (a == []):
        # So we don't crash accessing a[0]
        print([])
        return
    rows = len(a)
    cols = len(a[0])
    fieldWidth = maxItemLength(a)
    print("[ ", end="")
    for row in range(rows):
        if (row > 0): print("\n  ", end="")
        print("[ ", end="")
        for col in range(cols):
            if (col > 0): print(", ", end="")
            # The next 2 lines print a[row][col] with the given fieldWidth
            formatSpec = "%" + str(fieldWidth) + "s"
            print(formatSpec % str(a[row][col]), end="")
        print(" ]", end="")
    print("]")


print(solvePuzzle(board))#(getPossibleSpots(board))#isLegalSudoku(board)) 

