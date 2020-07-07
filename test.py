import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv",sep = ";")
# Since our data is seperated by semicolons we need to do sep=";"




# train our data, set them to what I want. There are 33 attributes in total for every student and we do not need that many.

print(data.head())
# This will print out the first 5 students in our data frame



data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# our data frame only has the information associated with those 6 attributes.



print(data.head())
# This will print out the first 5 students in our data frame which is now having only wanted attributes.



predict = "G3"
# This is also called as label; whatever you want in outcome



X = np.array(data.drop([predict], 1))
# X is all our training data, the ATTRIBUTES
# pandas.DataFrame.drop
# Drop specified labels from rows or columns.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
# Numpy provides a high-performance multidimensional array and basic tools to compute with and manipulate these arrays.
# SciPy builds on this, and provides a large number of functions that operate on numpy arrays and are useful for different types of scientific and engineering applications.

y = np.array(data[predict])
# our outcome data, the LABELS
# pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
# built on top of the Python programming language.



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.1)
# Here we take all our attributes and labels(X and y) and divide them into 4 arrays
# Split arrays or matrices into random train and test subsets
# we need to split our data into testing and training data. We will use 90% of our data to train and the other 10% to test.
# The reason we do this is so that we do not test our model on data that it has already seen.
# Scikit-learn is a free software machine learning library for the Python programming language.
# It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.


"""
linear = linear_model.LinearRegression()
# Model creation

linear.fit(x_train, y_train)
# https://scikit-learn.org/stable/modules/linear_model.html
acc = linear.score(x_test, y_test)
print(acc)
"""


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


"""

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)
        f = open("best _accuracy.txt", "w")
        f.write(str(best))
        f.close()
"""
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)



print('Coefficient: \n', linear.coef_)
# These are each slope value
print('Intercept: \n', linear.intercept_)
# This is the intercept



predictions = linear.predict(x_test)
# Gets a list of all predictions

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])


"""
# We save out model coz sometimes after training it saves upto like 90% accuracy. We will like that.
# Also if training a model takes hours, we will like to save that result and not train every time
with open("studentgrades.pickle", "wb") as f:
    pickle.dump(linear, f)
# pickle saves model for us
# linear is the name of the model we created above
# it should be defined above this...


pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)
# read the pickle file
# Now we can use linear to predict grades like before

# After dumping and saving the model, we can comment it out. Coz we have saved it to a model named linear again, we can continue to use it as before
"""

# Drawing and plotting model
p = "G1"
# or p is any feature
style.use("seaborn-colorblind")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()