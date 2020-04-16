import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as pyp
import numpy as np
from numpy import pi

#pandas information from the website: https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/
#matplotlib information from: https://matplotlib.org/users/pyplot_tutorial.html
#numpy information: https://docs.scipy.org/doc/numpy/user/quickstart.html 

nursesGlovesDF = pd.read_csv("Nurses gloves data - Sheet1 (1).csv", index_col = 0)
#nursesGlovesDF.info()

cols = nursesGlovesDF.columns
#print(cols)

nursesGlovesDF.rename(columns = {
            "Years Experience" : "yrsExp",
            'Before training' : "Before",
            '1 month after training' : "1 month",
            '6 months after training' : "6 months"
            }, inplace = True)
exp = nursesGlovesDF['yrsExp']
yrsExpRepeats = exp.value_counts()
#print(yrsExpRepeats)

def experience(s):
    if s == "4 or more":
        return "≥ 4"
    else:
        return "< 4"

nursesGlovesDF["yrsExp"] = nursesGlovesDF["yrsExp"].apply(experience)

nursesGlovesDF.boxplot(column='Before', by='yrsExp')

nursesGlovesDF.boxplot(column='1 month', by='yrsExp')
pyp.figure(3)
sub1 = pyp.subplot(111)
pyp.plot(nursesGlovesDF['6 months'], nursesGlovesDF['1 month'], "ro")
    #title='Change in cloves worn after 5 months')
m, b = np.polyfit(nursesGlovesDF['6 months'], nursesGlovesDF['1 month'], 1)
np.set_printoptions(precision=2)
lineArray = np.array([m, b])
#print(lineArray)
pyp.plot(nursesGlovesDF['6 months'], m*nursesGlovesDF['6 months'] + b, 'g-', 
        label=f'$y = {lineArray[0]}x + {lineArray[1]}')
sub1.legend(loc='upper left', bbox_to_anchor=(0, 1.00), shadow=False, ncol=1)

#plotting with made up data
pyp.figure(4)
data = {'a': np.sin(np.linspace(0, 2*pi, 100)),
        'colour': np.random.randint(0, 50, 100),
        'size': np.abs(np.random.randn(50))*100}
data['b'] = data['a'] + 100* np.random.randn(100)

pyp.scatter('a', 'b', c='colour', s='size', data=data)
pyp.xlabel('A')
pyp.ylabel('B')
pyp.show()
