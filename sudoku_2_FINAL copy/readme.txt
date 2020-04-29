Discription:

"Sudoku 2" is a game of sudoku puzzles that includes a neural network that can read a handwritten sudoku board. It can take get new puzzles for you to play using web scraping, solve the puzzles that you can't using backtracking, and give you a text file of your current board. 


How to run the project: 

The files neuralNetDigits.py, cmu_112_graphics.py, backtracking.py, isLegalSudoku.py useraInterface.py, and puzzleWebScraping.py should all be in the same folder. 

Run useraInterface.py. It may take take a couple seconds to load, but a menu window should pop up. 

Press 'i' to get instructions on how to play and the controls for the basic game. 
Press 'q' to get instructions on how to upload your own board to the game. 
If you lose the game (you can lose at any time by pressing 'enter' on the game screen to signal that you are done playing before you finish filling out the board), you can press 's' to see the board's solution.
If you don't like the board that you got, you can press 'r' to get a new game board.
If you press 'c', a txt file of your game board will be created in the game folder called 'yourText.txt'. 
Press 'p' to check if the current state of the game board is valid. 
Press 'enter' on the menu screen to begin playing. 


The libraries and modules used:

numpy, matplotlib, and pandas (data processing)
PIL (images)
requests and beautiful soup (bs4) (web scraping)

All libraries and modules that I used came with anaconda, which can be installed at https://docs.anaconda.com/anaconda/install/.


Shortcut commands:

There are no real shortcut commands. One thing that I do have is a file called "sampleBoard.png" included in the zip file that can be used to test the neural network without having to upload your own file. You are, of course, welcome to use your own file instead. 