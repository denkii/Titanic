# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the train variable.
import pandas as pd
import numpy as np
import os

# Import training data
os.chdir("C:\\Users\\bjkwok\\Documents\\Personal - Not Backed Up - Aucune sauvegarde\\References\\Titanic")
train = pd.read_csv("train.csv")

# Print the first 5 rows of the dataframe.
print(train.head(5))

# Fill in median age for rows with NA age
train["Age"] = train["Age"].fillna(train["Age"].median())

# Column characteristics
print(train.describe())

# Find all the unique genders
print(train["Sex"].unique())

# Encode Sex variable
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

# Find all the unique values for "Embarked".
print(train["Embarked"].unique())

# Encode Embarked variable
train.loc[train["Embarked"].isnull(), "Embarked"] = 0
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

# Import the linear regression and kfold class 
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

# Extract predictor columns
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the train dataset
kf = KFold(train.shape[0], n_folds=3, random_state=1)

pred = []
for train_i, test_i in kf:
    train_x = train[predictors].iloc[train_i,:]
    train_y  = train["Survived"].iloc[train_i]
    alg.fit(train_x,train_y)
    test_pred = alg.predict(train[predictors].iloc[test_i,:])
    pred.append(test_pred)

# Concatenate predictions from the numpy arrays
pred = np.concatenate(pred,axis=0)

# Use 0.5 decision boundary
pred[pred>0.5] = 1
pred[pred<=0.5] = 0

# Calculate accuracy for k-fold cross validation
print((train["Survived"] == pred).mean())

# Import logistic regression and cross validation
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# Init logistic regression algo
alg = LogisticRegression(random_state=1)

# Compute accuracy for folds
score = cross_validation.cross_val_score(alg,train[predictors],train["Survived"],cv=3)
print(score.mean())

# Import test.csv and perform same data manipulation as in training set
test = pd.read_csv("test.csv")
test["Age"] = test["Age"].fillna(train["Age"].median())
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
test.loc[test["Embarked"].isnull(), "Embarked"] = 0
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

# Fill missing Fare values in test set
test["Fare"] = test["Fare"].fillna(train["Fare"].median())

# Train and predict on logistic regression
alg.fit(train[predictors],train["Survived"])
pred = alg.predict(test[predictors])

# Create submission dataframe
output = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })

# Export to csv
output.to_csv("titanic-lr.csv",index=0)

