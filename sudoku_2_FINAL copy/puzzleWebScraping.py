#requests and beutiful soup came with anacoda, which I downloaded from: https://docs.anaconda.com/anaconda/install/
import requests
from bs4 import BeautifulSoup

#all the puzzles are scraped from http://www.menneske.no/sudoku/eng/ 
#This function gets the current sudoku puzzle from the webpage 
#the puzzle changes every time you reload (or request) the page
def getSudoku():
    numbers = ["1","2","3","4","5",'6','7',"8","9"]
    response = requests.get("http://www.menneske.no/sudoku/eng/")
    #using beautiful soup to get all the htlm tagged with 'tr'
    soup = BeautifulSoup(response.text, 'html.parser')
    boardHTML = soup.findAll('tr')[6:] #slice at 6 because that's where the board starts
    board = []
    for row in boardHTML:
        boardRow = []
        for col in row:
            currNum = 0
            colString = str(col)
            #figuring out which lines are actually cols and converting them to real numbers
            if "td" in colString:
                for number in numbers:
                    if number in colString:
                        currNum = int(number)
                boardRow.append(currNum)
        board.append(boardRow)
    return (board)
