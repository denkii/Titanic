# Titanic XGBoost
# Load libraries
library(plyr)
library(dplyr)
library(boot)
library(glmnet)
library(caret)
library(gbm)
library(e1071)
library(xgboost)
library(rpart)

# Import test and train data
setwd("C:/Users/bjkwok/Documents/Personal - Not Backed Up - Aucune sauvegarde/References/Titanic")
train = read.csv("train.csv")
test = read.csv("test.csv")
test$Survived = NA

# Combine datasets
data = rbind(train,test)
data$Survived = as.integer(data$Survived)

# Replace NA Embarked with "S" and factorise
data$Embarked[is.na(data$Embarked)] = "S"
data$Embarked = factor(data$Embarked)

# Replace missing Fare value with median
data$Fare[is.na(data$Fare)] = median(data$Fare, na.rm = T)

# Create new family_size variable
data$family_size = data$SibSp + data$Parch + 1

# Create new Title feature
data$Name = as.character(data$Name)
data$Title = factor(substr(data$Name, regexpr(',', data$Name) + 2, regexpr('\\.', data$Name) - 1))

# Use decision tree to predict Age withh ANOVA method for continuous variable
pred_age = rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,
		     data = data[!is.na(data$Age),], method = "anova")
data$Age[is.na(data$Age)] = predict(pred_age, data[is.na(data$Age),])

# Remove unnecessary variables
PassengerId = test$PassengerId
removeNames = c("PassengerId", "Name", "Ticket")
data = data[,!(names(data) %in% removeNames)]

# Split data back into train and test sets
train = data[!is.na(data$Survived),]
test = data[is.na(data$Survived),]

# Set predictors X and outcome y
outcomeName = "Survived"
predictorsNames = names(train)[names(train) != outcomeName]
y = train[,outcomeName]
X = data.matrix(train[,predictorsNames], rownames.force = NA)

# Remove outcome from data
train$Survived = NULL
test$Survived = NULL

# Create DMatrix for XGB
Dtrain = xgb.DMatrix(data = X, label = y, missing = NaN)
watchlist = list(x = Dtrain)

# Set seed for reproducibility
set.seed(123)

#xgboost fitting with optimal param
param = list(
	objective = "binary:logistic",
	booster = "gbtree",
	eval_metric = "error",
	eta = 0.4,
	max_depth = 3,
	subsample = 0.40,
	colsample_bytree = 0.40
)

#fit model with params above
xgb = xgb.cv(
	data = X,
	label = y,
	params = param,
	nrounds = 1000,
	watchlist = watchlist,
	verbose = F,
	maximize = F,
	nfold = 10,
	print.every.n = 1,
	early.stop.round = 10
)

#identify optimal nrounds
bestRound = which.min(as.matrix(xgb)[,3])

#train xgboost model
xgb = xgb.train(
	data = Dtrain,
	params = param,
	nrounds = bestRound,
	watchlist = watchlist,
	verbose = 1,
	maximize = F
)

#set test df as matrix
test = data.matrix(test, rownames.force=NA)

#predict and output to csv
Survived = predict(xgb, test)
output = data.frame(PassengerId, Survived)
output$Survived[output$Survived > 0.5] = 1
output$Survived[output$Survived <= 0.5] = 0
write.csv(output, "titanic-xgb.csv",row.names=F)