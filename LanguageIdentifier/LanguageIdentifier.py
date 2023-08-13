
"""
Created on Fri Feb 10 08:25:39 2023

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#Test data taken from online
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")


#Praparing the data for further processing 
#Extract content from text column
x = np.array(data["Text"])
#Extract content from language column
y = np.array(data["language"])

#Initialising vectorizer and converting text into numerical features 
cv = CountVectorizer()
X = cv.fit_transform(x)
#Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Multinomial Naive Bayes model
model = MultinomialNB()
#Training model
model.fit(X_train,y_train)
model.score(X_test,y_test)

#Get user input
userInput1 = input("Enter a Text: ")

#Counts the number of words the user has input and asks for more if sample is too small to give accurate answer
words = len(userInput1.split()) 
if words < 2 :
    print("Entered ", words, " word, please enter more for more accurate result")
elif words >= 2:
    print("Entered ", words, " words, please enter more for more accurate result")

#Transform user input into numerical data with same CountVectorizer 
data = cv.transform([userInput1]).toarray()
#Predict language of the user input using trained model
output = model.predict(data)
#Prints the predicted language 
print(output)

