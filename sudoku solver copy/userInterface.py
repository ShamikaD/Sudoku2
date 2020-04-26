from neuralNetDigits import runNet
from backtracking import isLegalSudoku, solvePuzzle
#graphics from: http://www.cs.cmu.edu/~112/notes/notes-animations-part1.html
from cmu_112_graphics import *
#numpy came with anacoda, which I downloaded from: https://docs.anaconda.com/anaconda/install/
import numpy as np
import random, copy, os, math

################################################################################
#                                                                              #
# ToDo:                                                                        # 
# 1. figure out what to do if the user inputs something that is not a 9X9 board#
# 2. impliment webscraping to find new puzzles                                 #
# 3. make buttons for all the options on the menu screen if I have time        #
#                                                                              #
################################################################################

#sets up for the board and the state of the game
def appStarted(app):
    app.margin = 10
    app.board = getDefaultBoard()
    app.rows = len(app.board)
    app.cols = len(app.board[0])
    app.cellHeight = (app.height-2*app.margin)/app.rows
    app.cellWidth = (app.width-2*app.margin)/app.rows
    app.boardColors = getBoardColors(app)
    app.state = "menu"
    app.isCorrecting = False
    app.currCell = (-1,-1)

#the default board that is used when nothing is inputted by the user
def getDefaultBoard():
    board = [[ 0, 0, 4, 1, 0, 0, 0, 9, 2],
            [ 3, 9, 8, 2, 5, 4, 0, 0, 7 ],
            [ 0, 0, 0, 0, 0, 0, 8, 3, 0 ],
            [ 0, 0, 0, 8, 7, 0, 0, 0, 0 ],
            [ 5, 1, 0, 4, 0, 2, 0, 8, 9 ],
            [ 0, 0, 0, 0, 3, 5, 0, 0, 0 ],
            [ 0, 5, 6, 0, 0, 0, 0, 0, 0 ],
            [ 9, 0, 0, 6, 4, 7, 2, 5, 8 ],
            [ 8, 7, 0, 0, 0, 9, 6, 0, 0 ]]
    return board

#colours the board so that playable squares are in red and set ones are black
def getBoardColors(app):
    boardColors = []
    for row in app.board:
        rowColors = []
        for col in row:
            if col == 0:
                rowColors.append("red")
            else:
                rowColors.append("black")
        boardColors.append(rowColors)
    return boardColors

#handles all the keypresses of the user
def keyPressed(app, event):
    key = event.key
    if key == 'm' and app.state != "win" and app.state != "lose":
            app.state = "menu"
    elif key == 'r' and (app.state == "win" or app.state == "menu" or 
        app.state == "lose" or app.state == "playable"): 
        appStarted(app)
    elif app.state == "menu":
        keyPressedMenu(app, key)
    elif app.state == "gameBoard":
        keyPressedGameBoard(app, key)
    elif app.state == "NNinstruct":
        keyPressedNNinstruct(app, key)
    elif key == 's' and (app.state == "lose" or app.state == "playable"):
        app.board = solvePuzzle(app.board)
        app.state = "gameBoard"

#handles all the keypresses when the user is on the gameboard screen
def keyPressedGameBoard(app, key):
    numbers = ["0","1","2","3","4","5","6","7","8","9"]
    if key == 'Enter':
        if app.isCorrecting:
            app.isCorrecting = False
            app.state = 'menu'
        else:
            if isLegalSudoku(app.board) and 0 not in np.array(app.board).flatten():
                app.state  = "win"
                app.currCell = (-1,-1)
            else:
                app.state = "lose"
                app.currCell = (-1,-1)
    elif key in numbers and app.currCell != (-1,-1):
        app.board[app.currCell[0]][app.currCell[1]] = int(key)

#handles all the keypresses when the user is on the menu screen
def keyPressedMenu(app, key):
    if key == 'Enter':
        app.state = "gameBoard"
    elif key == 'q':
        app.state = "NNinstruct"
    elif key == 'i':
        app.state = "instructions"
    elif key == 'p':
        app.state = "playable"

#handles all the keypresses when the user is on the neural network screen
def keyPressedNNinstruct(app, key):
    if key == 'c':
        txtFile = open("yourText.txt", "w")
        s = ""
        for row in app.board:
            for col in row:
                s+=str(col)
            s+="\n"
        txtFile.write(s)
        txtFile.close()
    elif key == 'f':
        app.isCorrecting = True
        app.state = "gameBoard"
    elif key == 't':
        fileName = app.getUserInput('Enter file name: ')
        if (os.path.exists(fileName)):
            app.board = runNet(fileName)
            app.boardColors = getBoardColors(app)

#gets the coordinates on the grid of the x and y position of a mouse press
def getCellCoords(app, x, y):
    r = int((y - app.margin) // app.cellHeight)
    c = int((x - app.margin) // app.cellWidth)
    return (r, c)

#allows the user to select where they are going to change a number
def mousePressed(app, event):
    x = event.x
    y = event.y
    if app.state == "gameBoard":
        #checks if the area clicked is within thw bounds of the playable area
        if (x > app.margin and x < app.width-app.margin and
            y > app.margin and y < app.height-app.margin):
            col, row = getCellCoords(app, x ,y)
            if app.isCorrecting == True or app.boardColors[row][col] == 'red':
                app.currCell = (row, col)
            else:
                app.currCell = (-1,-1)
        else:
            app.currCell = (-1,-1)

#Keeps updating the cell height and width so that they can be resized with the window
#also keeps updating the colours of the board when the user is correcting it
def timerFired(app):
    app.cellHeight = (app.height-2*app.margin)/app.rows
    app.cellWidth = (app.width-2*app.margin)/app.rows
    if app.isCorrecting:
        app.boardColors = getBoardColors(app)
    
#draws the canvas for when the user is playing the game
def drawGameBoard(app, canvas):
    canvas.create_rectangle(0,0,app.width, app.height)
    lastRow = app.rows * app.cellWidth + app.margin
    lastCol = app.cols * app.cellHeight + app.margin
    canvas.create_rectangle(lastRow, app.margin, lastRow+1, app.height - app.margin, 
        fill = "black")
    canvas.create_rectangle(app.margin, lastCol-1, app.width - app.margin, lastCol+1, 
        fill = "black")
    for r in range (app.rows):
        for c in range(app.cols):
            row = r * app.cellWidth + app.margin
            col = c * app.cellHeight + app.margin
            #draws the lines on the board
            canvas.create_rectangle(row, col, row + app.cellWidth, 
                col+app.cellHeight)
            if r% math.sqrt(app.rows) == 0:
                canvas.create_rectangle(row-1, app.margin, row+1, app.height - app.margin, 
                    fill = "black")
            if c% math.sqrt(app.cols) == 0:
                canvas.create_rectangle(app.margin, col-1, app.width - app.margin, col+1, 
                    fill = "black")
            #Draws the grid
            canvas.create_text(row + 0.5*app.cellWidth, col + 0.5*app.cellHeight, 
                text = str(app.board[r][c]), font='Arial '+str(int(app.cellHeight/2)), 
                    fill = app.boardColors[r][c])
            #draws the numbers
            if r == app.currCell[0] and c == app.currCell[1]:
                canvas.create_text(row + 0.5*app.cellWidth, col + 0.5*app.cellHeight, 
                    text = str(app.board[r][c]), font='Arial '+str(int(app.cellHeight/2)), 
                        fill = "green")

#draws the canvas for when the user is looking at the unstructions for using the neural network
def drawNNinstruct(app, canvas):
    canvas.create_text(app.width/2,0, 
        text = "Instructions", font='Silom '+str(int(app.height/10)) + " bold", 
            anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 1.5*app.cellHeight, 
        text = "Press 'm' at any time to return to the menu", font='Arial '
            +str(int(app.height/30)) + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 2.5*app.cellHeight, 
        text = "Put an image of your board in the game folder.", font='Arial '
            +str(int(app.height/30)) + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 3.5*app.cellHeight, 
        text = "Press 't' to enter your file name", font='Arial '
            +str(int(app.height/30)) + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 4.5*app.cellHeight, 
        text = "Press 'c' if you would like a copy of the", font='Arial '
            +str(int(app.height/30)) + " bold", anchor="n", fill = "purple") 
    canvas.create_text(app.width/2, 5*app.cellHeight, 
        text = "numbers saved as a text file", font='Arial '+str(int(app.height/30)) 
            + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 6*app.cellHeight, 
        text = "Press 'f' to change any numbers that the ", font='Arial '
            +str(int(app.height/30)) + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 6.5*app.cellHeight, 
        text = "network might have read wrong", font='Arial '
            +str(int(app.height/30)) + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 7.5*app.cellHeight, 
        text = "Press 'Enter' when you are done correcting", font='Arial '
            +str(int(app.height/30)) + " bold", anchor="n", fill = "purple")

#draws the canvas for when the user is looking at the instructions
def drawInstructions(app, canvas):
    canvas.create_text(app.width/2,0, 
        text = "Instructions", font='Silom '+str(int(app.height/10)) + " bold",
            anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 1.5*app.cellHeight, 
        text = "Press 'm' at any time to return to the menu", font='Arial '
            +str(int(app.height/30)) + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 2.5*app.cellHeight, 
        text = "To Play The Game:", font='Arial '+str(int(app.height/30)) 
            + " bold", anchor="n", fill = "purple")
    canvas.create_text(10, 3*app.cellHeight, 
        text = "Click on a number to select it. ", font='Arial '+str(int(app.height/35)) 
            + " bold", anchor="nw", fill = "purple")
    canvas.create_text(10, 3.5*app.cellHeight, 
        text = "Then type in the number you would like to enter.", font='Arial '
            +str(int(app.height/35)) + " bold", anchor="nw", fill = "purple")
    canvas.create_text(10, 4*app.cellHeight, 
        text = "Once you are sure of your solution, press 'Enter' to check if you won!", 
            font='Arial '+str(int(app.height/35)) + " bold", anchor="nw", fill = "purple")
    canvas.create_text(app.width/2, 5.5*app.cellHeight, 
        text = "Rules Of The Game:", font='Arial '+str(int(app.height/30))+ " bold",
            anchor="n", fill = "purple")
    canvas.create_text(10, 6*app.cellHeight, 
        text = "To win, change red numbers so that all the numbers from 1 to 9 are", 
            font='Arial '+str(int(app.height/35)) + " bold", anchor="nw", fill = "purple")
    canvas.create_text(10, 6.5*app.cellHeight, 
        text = "in but never repeat in every row, column, diagonal, and outlined square.", 
            font='Arial '+str(int(app.height/35)) + " bold", anchor="nw", fill = "purple")

#draws the canvas for when the user is on the menu 
def drawMenu(app, canvas):
    canvas.create_text(app.width/2,0, 
        text = "Sudoku Solver", font='Silom '+str(int(app.height/10)) + " bold", 
            anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 1.5*app.cellHeight, 
        text = "Press 'r' to restart", font='Arial '+str(int(app.height/30)) + " bold", 
            anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 2.5*app.cellHeight, 
        text = "Press 'm' to return to the menu at any time", font='Arial '+str(int(app.height/30)) 
            + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 3.5*app.cellHeight, 
        text = "Press 'q' to enter your own game board", font='Arial '+str(int(app.height/30))
             + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 4.5*app.cellHeight, 
        text = "Press 'i' to see the instructions", font='Arial '+str(int(app.height/30)) 
            + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 5.5*app.cellHeight, 
        text = "Press 'p' to check if the current state of the board is winable ", 
            font='Arial '+str(int(app.height/30)) + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, 6.5*app.cellHeight, 
        text = "Press 'enter' to start playing", font='Arial '+str(int(app.height/30)) 
            + " bold", anchor="n", fill = "purple")

#draws the canvas for when the user is testing if the current board is playable
def drawPlayable(app, canvas):
    isPlayable = isLegalSudoku(app.board)
    if isPlayable:
        playable = "Possible To Win"
    else:
        playable = "Impossible To Win"
    canvas.create_text(app.width/2,app.height/2, 
        text = playable, font='Silom '+str(int(app.height/10)) + " bold", 
            fill = "purple")
    canvas.create_text(app.width/2, app.height/2+ 0.5*app.cellHeight, 
        text = "Press 's' to see the solution", font='Arial '+str(int(app.height/30)) 
            + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, app.height/2+ 1.5*app.cellHeight, 
        text = "Press 'r' restart", font='Arial '+str(int(app.height/30)) 
            + " bold", anchor="n", fill = "purple")
    canvas.create_text(app.width/2, app.height/2+ 2.5*app.cellHeight, 
        text = "Press 'm' return to the menu", font='Arial '+str(int(app.height/30)) 
            + " bold", anchor="n", fill = "purple")

#draws the canvas depending on what state the game is in
def redrawAll(app, canvas):
    if app.state == 'gameBoard':
        drawGameBoard(app, canvas)
    elif app.state == 'menu':
        drawMenu(app, canvas)
    elif app.state == 'NNinstruct':
        drawNNinstruct(app, canvas)
    elif app.state == 'instructions':
        drawInstructions(app, canvas)
    elif app.state == 'win':
        canvas.create_text(app.width/2,app.height/2, 
            text = "YOU WON!", font='Silom '+str(int(app.height/10)) + " bold", 
                fill = "purple")
        canvas.create_text(app.width/2, app.height/2+ 0.5*app.cellHeight, 
            text = "Press 'r' to restart", font='Arial '+str(int(app.height/30)) 
                + " bold", anchor="n", fill = "purple")
    elif app.state == "lose":
        canvas.create_text(app.width/2,app.height/2, 
            text = "YOU LOST!", font='Silom '+str(int(app.height/10)) + " bold", 
                fill = "purple")
        canvas.create_text(app.width/2, app.height/2+ 0.5*app.cellHeight, 
            text = "Press 's' to see the solution", font='Arial '+str(int(app.height/30)) 
                + " bold", anchor="n", fill = "purple")
        canvas.create_text(app.width/2, app.height/2+ 1.5*app.cellHeight, 
            text = "Press 'r' to restart", font='Arial '+str(int(app.height/30))    
                + " bold", anchor="n", fill = "purple")
    elif app.state == "playable":
        drawPlayable(app, canvas)

runApp(width = 600, height = 600)
print("compiles :)")