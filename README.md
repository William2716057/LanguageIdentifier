# LanguageIdentifier

Data Preparation:

The script loads a dataset containing text data and language labels from an online source.
It extracts the text content and language labels into separate numpy arrays.
Text Vectorization:

The CountVectorizer is used to convert the text data into numerical features.
The dataset is split into training and testing sets using train_test_split.
Model Training:

A Multinomial Naive Bayes model is initialized using MultinomialNB().
The model is trained on the training data using the fit method.
User Input and Prediction:

The user is prompted to input a text.
The script counts the number of words in the input and provides feedback based on the word count.
The user input is transformed into numerical data using the same CountVectorizer.
The trained model predicts the language of the user input.
The predicted language is printed to the console.
Dataset
The test data used in this script is taken from this online source.

Requirements
Python 3.x
pandas
numpy
scikit-learn
