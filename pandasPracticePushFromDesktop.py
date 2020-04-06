import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as pyp
import numpy as np
import json
import os

#mandas documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
#pandas information from the website: https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/
'''
PANDAS:

series (columns) and dataframes (table made of series)
'''

#creating a data frame
data = {
            "Item a": [2,1,4],
            "Item b": [3,62,1],
            "Item c": [1,34,1]
        }
#passing to a Dataframe constructor
purchases = pd.DataFrame(data)

#now, purchases will print to a cool table where keys are labels 
# and values are the values under that label
#print(purchases)

#to make labels for each col, add "index"
purchases2 = pd.DataFrame(data, index = ["Bob", "Susan", "Harold"])
#print(purchases2)

#because the cols are labled, we can now use "loc" to search by those labels 
SusanPurchases = purchases2.loc["Susan"]
#print(SusanPurchases)

#Reading Data from a csv
nursesGlovesDF = pd.read_csv("Nurses gloves data - Sheet1.csv", index_col = 0)
#print(nursesGlovesDF)

#Reading Data from a JSON file
data = json.load(open("sampleDataHawai.json"))
hawaiiSampleDF = pd.DataFrame(data["data"]) #argument to orient the json file 
#print(hawaiiSampleDF)

#Saving your data from python to CSV or JSON file
#nursesGlovesDF.to_json("nursesGlovesDF.json")
#hawaiiSampleDF.to_csv("hawaiiSampleDF.csv")

#to preview your data
nurse2 = nursesGlovesDF.head(2) #shows the first 2 rows of the nursesGlovesDF data frame
#print(nurse2)
nurse3 = nursesGlovesDF.tail(3) #shows the last 3 rows of the nursesGlovesDF data frame
#print(nurse3)

#This is how you get an overview of info from your file
#   This should be run before you try doing things to your data so you can see
#    what youre dealing with. It also prints without a print statement
#nurseInfo = nursesGlovesDF.info()

#"shape" gives the rows and cols of the data that in your dataframe as a tuple
nurseSize = nursesGlovesDF.shape
#print(nurseSize)

#to combine two dataframes use "append()". This is non-destructive
new = nursesGlovesDF.append(hawaiiSampleDF)
#print(new)
#print(nursesGlovesDF)

#To get rid of duplicate rows in your data (non-destructive)
#print(nursesGlovesDF.shape)
noDup = nursesGlovesDF.drop_duplicates()
#print(noDup.shape)

#different ways to remove duplicates
    #takes out all duplicates except the first instance (default if no keep is written)
noDup = nursesGlovesDF.drop_duplicates(keep = "first") 
    #takes out all duplicates except the lasr instance
noDup = nursesGlovesDF.drop_duplicates(keep = "last")
    #takes out all duplicates, like, all of them
noDup = nursesGlovesDF.drop_duplicates(keep = False)

#to make methods destructive, use inplace = True (comes before "keep" as an argument)
#print(nursesGlovesDF.shape)
nursesGlovesDF.drop_duplicates(inplace = True)
#print(nursesGlovesDF.shape)

#getting a list of the column names in your df
cols = nursesGlovesDF.columns
#print(cols)

#renaming cols using 
#method 1: manually
nursesGlovesDF.rename(columns = {
            "Years Experience" : "yrs exp",
            'Before training' : "before",
            '1 month after training' : "1_month",
            '6 months after training' : "6_months"
            }, inplace = True)
#print(nursesGlovesDF.columns)
#method 2: list comprehension
nursesGlovesDF.columns = [col.lower() for col in nursesGlovesDF]

#dealing with null values
    #step 1: finding out what is null / where the nul values are 
nullDFNurse = nursesGlovesDF.isnull() #returns a dataframe of booleans
#print(nullDFNurse)
nulColCountDurse = nursesGlovesDF.isnull().sum() #returns the number of null values per col
#print(nulColCountDurse)

    #step 2: getting rid of the nul vals
        #option 1: delete any row with a null value in it (non-destructive)
noNull = nursesGlovesDF.dropna() 
        #option 2: delete any cols with a nul value in them (non-destructive)
noNull = nursesGlovesDF.dropna(axis = 1) 
        #option 3 (usually best): replave any nul values with something else
            #Step 1: take out the col that you want to replace a val in
month1 = nursesGlovesDF['1_month']
            #Step 2: find what youre going to replace with (usually mean)
month1Mean = month1.mean()
            #Step 3: fill with that value
month1.fillna(month1Mean, inplace = True)

#to get a summary of distribution 
descript = nursesGlovesDF.describe()
#print(descript)

#to find the amount that a value repreats in a row
month1Repeats = month1.value_counts()
#print(month1Repeats)

#to find the relationship between continuous variables: makes it easy to find trends
relat = nursesGlovesDF.corr()
#print(relat)

#you can iterate over a dataframe like a list, but that is slow
#use functions instead

def enoughGlovesFunction(n):
    if n == 3:
        return "yes"
    else:
        return "no"

nursesGlovesDF["1_month"] = nursesGlovesDF["1_month"].apply(enoughGlovesFunction)
#print(nursesGlovesDF)

#The fun stuff: plotting! with matplotlib
#adjust the font and figure size
mpl.rcParams.update({'font.size': 20, 'figure.figsize': (10, 8)})

#now we can plot!
    #scatter plot
nursesGlovesDF.plot(kind='scatter', x='before', y='6_months', title='gloves worn 5 months after training')
pyp.show()
    #histogram
nursesGlovesDF.plot(kind='hist', title='gloves worn 5 months after training')
pyp.show() #this is not a good histogram because of the data set
    #box plot
nursesGlovesDF.plot(kind='box', title='gloves worn 5 months after training')
pyp.show()
    #box plot sorted by the words in the 1_month catagory
nursesGlovesDF.boxplot(column='6_months', by='1_month')
pyp.show()

#For categorical variables: Bar Charts* and Boxplots.
#For continuous variables: Histograms, Scatterplots, Line graphs, and Boxplots.

#more matplotlib stuff! 
#from: https://matplotlib.org/users/pyplot_tutorial.html
