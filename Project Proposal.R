#################################
##  Data Incubator Challenge   ##
################################

## Download dataset 
setwd("C:/Documents and Settings/128267/Desktop/Coursera/data analysis/Data Analysis Assignment 2")

fileURL<-"https://spark-public.s3.amazonaws.com/dataanalysis/samsungData.rda"
download.file(fileURL, destfile="Samsungdat.rda")
load("Samsungdat.rda")

head(samsungData)
tail(samsungData)

###########################

str(samsungData)
summary(samsungData)
names(samsungData)

table(samsungData$subject)
table(samsungData$activity)

## Fix the variable name issues
oldnames <- names(samsungData) 
samsungData <- data.frame(samsungData)
samsungData$activity <- as.factor(samsungData$activity)

#### Change the variable names 

library(reshape)
samsungData <- rename(samsungData, c(fBodyAccJerk.std...X="fBodyAccJerk.SD"))
samsungData <- rename(samsungData, c(tGravityAcc.mean...X="tGravityAcc.Mean"))
samsungData <- rename(samsungData, c(angle.Y.gravityMean.="Angle"))
samsungData <- rename(samsungData, c(tBodyAcc.max...X="tBodyAcc.max"))
samsungData <- rename(samsungData, c(fBodyAccJerk.std...X="fBodyAccJerk.SD"))
samsungData <- rename(samsungData, c(tGravityAcc.min...Y="tGravityAcc.min"))
samsungData <- rename(samsungData, c(tGravityAcc.arCoeff...Z.4="tGravityAcc.arCoeff"))
samsungData <- rename(samsungData, c(tGravityAcc.energy...Y="Energy"))
samsungData <- rename(samsungData, c(tBodyAccJerk.max...X="tBodyAccJerk.max"))


## Split the dataset to training set and test set 

### Training set include subject 1,3,5,6,7,8,11,14
SamsungTrain <- samsungData[samsungData$subject %in% c(1,3,5,6,7,8,11,14), ]

### Test set include subject 23,25,26,27,28,29,30
SamsungTest <- samsungData[samsungData$subject %in% c(23,25,26,27,28,29,30), ]


#################################################

### Build a tree model to the training set 

#install.packages('tree')

library(tree)
tree.train <- tree(activity ~ ., data=SamsungTrain)
summary(tree.train)

par(mfrow=c(1,2))

plot(tree.train)
text(tree.train)

### Prune the tree - cross-validation (k-fold)

par(mfrow=c(1,2))

plot(cv.tree(tree.train, FUN=prune.tree, method="misclass"))
plot(cv.tree(tree.train))

###################################
###################################

par(mar=c(4,4,6,2))
plot(cv.tree(tree.train, FUN=prune.tree, method="misclass", K=5), 
     main="K-fold Cross-Validation on the Training Set")
abline(v=8,col=2)

###################################
###################################

###################################
###################################

pruneTree1<-prune.tree(tree.train, best=8)
plot(pruneTree1)
text(pruneTree1, digits=2)
title("Decision Tree Model to Predict Physical Activities")

###################################
###################################

summary(pruneTree1)

# [1] "BodyAccJerk.SD"      "tGravityAcc.Mean"    "Angle"               "tBodyAcc.max"       
# [5] "tGravityAcc.min"     "tGravityAcc.arCoeff" "tGravityAcc.energy" 
# Number of terminal nodes:  8 
# Residual mean deviance:  0.4966 = 1259 / 2535 
# Misclassification error rate: 0.07118 = 181 / 2543 


## Cross-validation suggests 8 variables 

par(mfrow=c(1,1))

pruneTree0<-prune.tree(tree.train, best=10)
plot(pruneTree0)
text(pruneTree0)

pruneTree2<-prune.tree(tree.train, best=6)
plot(pruneTree2)
text(pruneTree2)

### Show substitution error ####

## tree 0 
table(SamsungTrain$activity, predict(pruneTree0, type="class"))

## tree 1
table(SamsungTrain$activity, predict(pruneTree1, type="class"))


## tree 2
table(SamsungTrain$activity, predict(pruneTree2, type="class"))

#################################################################
#################################################################

### Predict new values

predict1<-predict(pruneTree1, SamsungTest, type="class")
predict1
summary(predict1)


##### Determine the resubstitution and prediction error rate

## Missclassification function
missClass = function(values,prediction){sum(prediction != values)/length(values)}


###### Tree model 0 ####

## Missclass rate for training set -- rate = 0.0660637
missClass(SamsungTrain$activity, predict(pruneTree0, SamsungTrain, type="class"))

## Missclass rate for testing set  -- rate = 0.1459744
missClass(SamsungTest$activity, predict(pruneTree0, SamsungTest, type="class"))


###### Tree model 1 ####
########################################
########################################

## Missclass rate for training set -- rate = 0.07117578
missClass(SamsungTrain$activity, predict(pruneTree1, SamsungTrain, type="class"))

## Missclass rate for testing set  -- rate = 0.1482318
missClass(SamsungTest$activity, predict(pruneTree1, SamsungTest, type="class"))

########################################
########################################

###### Tree model 2 ####

## Missclass rate for training set -- rate = 0.1104994
missClass(SamsungTrain$activity, predict(pruneTree2, SamsungTrain, type="class"))

## Missclass rate for testing set  -- rate = 0.2121896
missClass(SamsungTest$activity, predict(pruneTree2, SamsungTest, type="class"))



