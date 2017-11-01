#Set the working directory
setwd("/users/pranavdar/Desktop/PGPBABI/Finance and Risk Analytics/off campus assignment")

#Import the required libraries
library(readxl)
library(dplyr)
library(Amelia)
library(corrplot)
library(ggplot2)
library(mice)
library(DMwR)
library(randomForest)
library(caret)
library(pROC)
library(e1071)

#Import the training and testing dataset
loan_train <- read_excel("training.xlsx", 2)
loan_test <- read_excel("test.xlsx", 2)

str(loan_train)
summary(loan_train)
str(loan_test)
summary(loan_test)

########################## DATA CLEANING #####################################
#Shorten the variable names for ease of use
colnames(loan_train) <- c("case_no", "delq", "rev_util_unseclines", "debtratio", "no_creditline", 
                          "no_dep")
colnames(loan_test) <- c("case_no", "delq", "rev_util_unseclines", "debtratio", "no_creditline", 
                         "no_dep")

#Check for any missing values
any(is.na(loan_train))
any(is.na(loan_test))

#Convert to factor and integer
loan_train$delq <- as.factor(loan_train$delq)
loan_train$no_dep <- as.integer(loan_train$no_dep)
loan_test$delq <- as.factor(loan_test$delq)
loan_test$no_dep <- as.integer(loan_test$no_dep)

#As we can see, there were a number of missing values in the dependants column and we will
#need to impute them before proceeding
any(is.na(loan_train))
sum(is.na(loan_train))

any(is.na(loan_test))
sum(is.na(loan_test))

#Merge the 2 datasets for imputing values. We will split them again later
loan_default <- rbind(loan_train, loan_test)

#Take a look at the dataset
glimpse(loan_default)

#Check for the missing values
sum(is.na(loan_default))

#Double check graphically where the missing values are
missmap(loan_default, col = c("yellow", "black"))

#Dropping the case_no variable
loan_default <- loan_default %>% select(-1)

#Impute missing values
#We will use the MICE package and method to do this
md.pattern(loan_default)

init <- mice(loan_default, maxit=0)
meth <- init$method
predM <- init$predictorMatrix

#Skipping these variables from imputation and using them as predictors
meth[c("delq", "rev_util_unseclines", "debtratio", "no_creditline")] = ""

meth[c("no_dep")] = "pmm"

set.seed(101)
loan_default_imp <- mice(loan_default, method=meth, predictorMatrix = predM, m=5)

loan_default_imp <- complete(loan_default_imp)

#Check the database to see if the missing values have been imputed
md.pattern(loan_default_imp)
View(loan_default_imp)

#Split the data back into training and testing data as before
loan_train <- loan_default_imp[1:5000,]
loan_test <- loan_default_imp[5001:6000,]


##################### DATA EXPLORATION and VISUALIZATION ############################
#Check the number of 0's and 1's in the data
loan_default_imp %>% group_by(delq) %>% summarise(n=n())
loan_train %>% group_by(delq) %>% summarise(n=n())
loan_test %>% group_by(delq) %>% summarise(n=n())

#Explore the dataset using dplyr
loan_train %>% group_by(no_dep) %>% summarise(n=n_distinct(no_creditline))

#Correlation checks
m <- cor(loan_train[2:5])
corrplot(m)

#Visualize the data using the ggplot2 package
qqnorm(loan_train$debtratio)
loan_train <- loan_train[-4855,]

qqnorm(loan_train$no_creditline)

qqnorm(loan_train$rev_util_unseclines)

ggplot(loan_train, aes(no_creditline)) + geom_bar(aes(fill=delq))

ggplot(loan_train, aes(no_dep)) + geom_bar(aes(fill=delq))

ggplot(loan_train, aes(no_creditline)) + geom_histogram()

ggplot(loan_train, aes(no_dep)) + geom_histogram()

ggplot(loan_train, aes(debtratio)) + geom_histogram()

########################## PRE-MODEL BUILDING TASKS ###############################
#SMOTE technique
trainSMOTE <- SMOTE(delq ~ ., perc.over=500, perc.under=140, data=loan_train)

#Check the %age of 1s in the SMOTE'd data
table(loan_default$delq)
table(trainSMOTE$delq)

#Centring and Scaling
trans <- preProcess(trainSMOTE,method = c("center", "scale")); trans

train.centered <- predict(trans,trainSMOTE)

################################### Model Building and Predictions ################################

###################### Logistic Regression
loan_logit <- glm(delq ~ ., family=binomial, data=train.centered)
summary(loan_logit)

#Dropping the 'rev_util_unseclines' variable and running the model again
loan_logit <- glm(delq ~ debtratio + no_creditline + no_dep, family=binomial, data=train.centered)
summary(loan_logit)

#Predict using the logit model on the loan dataset
logit_score <- predict(loan_logit, newdata=loan_test, type="response")
fitted.score <- ifelse(logit_score > 0.3, 1, 0)

#Let's look at the Confusion Matrix
delq.logit <- confusionMatrix(fitted.score, loan_test$delq)
delq.logit

#Draw the ROC curve
roc.logit <- roc(loan_test$delq, fitted.score)
roc.logit
plot(roc.logit)

#################### Naive Bayes Model
nBayesfit <- naiveBayes(delq ~ ., data=train.centered, usekernel=T)

nb_score <- predict(nBayesfit, newdata=loan_test)

#Let's look at the Confusion Matrix
delq.nb <- confusionMatrix(nb_score, loan_test$delq)
delq.nb

################# Random Forest
#Create the random forest model
rf <- randomForest(delq ~ ., data=train.centered, mtry = 4, ntree = 101, importance = T)
plot(rf)
importance(rf)

#Predict using the rf model
rf_score <- predict(rf, newdata = loan_test)
table(rf_score, loan_test$delq)

################# SVM
#Create the SVM model
loan_svm <- svm(delq ~ ., data=train.centered)
summary(loan_svm)

#Predict using the SVM model
svm_score <- predict(loan_svm, newdata = loan_test)
table(svm_score, loan_test$delq)
