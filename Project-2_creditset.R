
# PROJECT: #To Build a Logistic Regression Model to predict default 10yr a categorical variable in creditset using the predictors age,income and loan using the creditset.csv

# Solution:
# Using logistic regression, predict the default 10yr consuming theother predictor variables in the dataset. Note the response variable is a categorical variable. As a solution strategy - split the dataset, train the model and test the model and conclude the final outcome of analysis

# loading creditset.csv into the R Environment and verify the dimensions
creditset<- read.csv(file.choose())
View(creditset)
dim(creditset)
names(creditset)

#splitting our data into training and testing datasets
library(caTools)

set.seed(2)
split<-sample.split(creditset, SplitRatio = 0.75); split
training<-subset(creditset, split =="TRUE");training
testing<-subset(creditset, split == "FALSE");testing

# Investigate the list of defaulters in each datasets - creditset, training & testng
table(creditset$default10yr) #displays total count of defaulters & non defaulters in creditset
table(training$default10yr) #displays total count of defaulters & non defaulters in training ds
table(testing$default10yr)  #displays total count of defaulters & non defaulters in testing ds

#default 10yr being a categorical variable, 1 represents defaulter and 0 represents non defaulter. Make use of logistic regression to predict the default10yr dependent variable

# above we have divided our data set into 2 parts  training and testing
# we are going to build our model /train our model with training dataset
# and will test the same model on our testing dataset
# training & testing are the subsets of creditset

# building our  Logistic Regression model
# as clientid  and LTI are not the predictors of default10yr we will not use these variables while building our model
# Our predictor variables are age,income and loan
# we need to decide and optimize our model in such a way that the accuracy is more
# and the threshold cutoff probability is such that the model is not dangerous
# like we predict defaulters as non-defaulters as such an info will be dangerous for bank
# meaning  False-Negative should be minimum but  not much accuracy of the model is lost

# Building Logistic Regression model using our  training dataset
model<-glm(default10yr~income+age+loan,data=training,family= binomial)
model

# Hence our logit function predictors are as below
# Constant term  is 9.8166458
# Coefficient for  Income is  -0.0002367
# Coefficient for  age  is  -0.3465840
# Coefficient for  loan is  0.0017076

# our Logit function becomes

# logit(odds) = logit(default10yr) =  9.8166458 + ((-0.0002367) * Income) + ((-0.3465840) * age) + ((0.0017076) *loan)

# P(default10yr being '1') = e^(9.8166458 + ((-0.0002367) * Income) + ((-0.3465840) * age) + ((0.0017076) *loan)) / ( 1+ e^(9.8166458 + ((-0.0002367) * Income) + ((-0.3465840) * age) + ((0.0017076) *loan))

# summary(model)

#Optimizing Our Model - Building others models to optimize "model" by trying combinations of predictors

model1<-glm(default10yr~income,data=training,family= binomial)
summary(model1)
model2<-glm(default10yr~age,data=training,family= binomial)
summary(model2)
model3<-glm(default10yr~loan,data=training,family= binomial)
summary(model3)
model4<-glm(default10yr~age+income,data=training,family= binomial)
summary(model4)
model5<-glm(default10yr~age+loan,data=training,family= binomial)
summary(model5)
model6<-glm(default10yr~income+loan,data=training,family= binomial)
summary(model6)

# In all the above models from model1 to model6 the AIC value has not decreased
# hence we cannot remove any of the predictors from Logistic Regression model named "model"
# So we can say all the three predictors age,income & loan are good predictors of default10yr
# our Final model for predicting default10yr is "model" which is optimized


#Predicting Probabilities for training dataset
library(CARS)
res<-predict(model,training,type="response") # for training dataset
head(res)

# Confusion Matrix for our results when compared with predicted values for training dataset
# we have threshold prob of 0.5 in this case which is taken by default
tabtraining0.5<-table(ActualValue=training$default10yr,PredictedValue=res>=0.5)
tabtraining0.5   #checked with probability cutoff = 0.5
accuracytraining0.5<-sum(diag(tabtraining0.5)/sum(tabtraining0.5))
accuracytraining0.5 #accuracy at 0.5 cutoff prob


#our model built is accurate to upto 95%  meaning it predicted defaulter as defaulter & non-defaulter as non-defaulter in the training dataset from creditset with almost 95% accuracy
# Predicting Probabilties for testing dataset

# now checking our model for accuracy with testing dataset
# Finding the predicted probabilties associated with observations using our model for testing dataset
res1<-predict(model,testing,type="response")  # for testing dataset
head(res1)

# Confusion Matrix for our results when compared with predicted values for testing dataset
tabtesting0.5<-table(ActualValue=testing$default10yr,PredictedValue=res1>=0.5)
tabtesting0.5 #checked with probability cutoff = 0.5
accuracytesting0.5<-sum(diag(tabtesting0.5)/sum(tabtesting0.5))
accuracytesting0.5 #accuracy at 0.5 cutoff prob


#our model built is accurate to upto 95%  meaning after verifying ith testing dataset from creditset.As it predicted & we verified defaulter as defaulter & non-defaulter as non-defaulter in the testing dataset from creditset with almost 95% accuracy

# ROC curve
# now we will try to check using ROCR package whether our cutoff Probability is with less False-Negative or not .
# We can also check using ROC Curve whether our cutoff probablity is most closest to NW corner of the graph or not meaning it has least distance with NW corner of graph .
library(ROCR)
ROCRPred<-prediction(res,training$default10yr)
ROCRPref<-performance(ROCRPred,"tpr","fpr")
plot(ROCRPref,colorize=TRUE,main="ROC Curve", print.cutoffs.at=seq(0.1,by=0.1))

# We can see that  the distance between the 0.4 probabilty and the Northwest corner is  less as compared to  0.5
# so we can decide to take our cutoff probabilty as 0.4
# this probability will also decrese number of false-Negatives which were dangerous to our model

#### Trying different Probabilty values as cutoff to check diffrent Confusion matrices
# checking confusion Matrix for diffrent Probabilities for our training dataset
table(ActualValue=training$default10yr,PredictedValue=res>=0.3) #checked with probability cutoff = 0.3
table(ActualValue=training$default10yr,PredictedValue=res>=0.4) #checked with probability cutoff = 0.4
table(ActualValue=training$default10yr,PredictedValue=res>=0.5) #checked with probability cutoff = 0.5

# so finally As per the model our probability cutoff is  0.4 as it gives me a good accuracy along with less false negatives
# Thus giving the Confusion matrix as

tabtraining0.4<-table(ActualValue=training$default10yr,PredictedValue=res>=0.4) #checked with probability cutoff = 0.4
tabtraining0.4

table(training$default10yr)
accuracytraining0.4<-sum(diag(tabtraining0.4))/sum(tabtraining0.4)
accuracytraining0.4

#Sensitivity & Specificity & Accuracy for training dataset
#calculating Sensitivity & Specificity for dataset training for  p=0.4
sensitivitytraining0.4<-tabtraining0.4[4]/(tabtraining0.4[2]+tabtraining0.4[4])
sensitivitytraining0.4

specificitytraining0.4<-tabtraining0.4[1]/(tabtraining0.4[1] + tabtraining0.4[3])
specificitytraining0.4

# Classification Accuracy for group 0 for training dataset is equal to  "specificitytraining0.4" calculated as shown in output

# Classification Accuracy for group 1 is equal to  calculated as "sensitivitytraining0.4" shown in  output

# Classification Accuracy of our model for training dataset is given by  equal to
classification_accuracy_training0.4<- (sensitivitytraining0.4+specificitytraining0.4)/2
classification_accuracy_training0.4

#Overall accuracy  found by using the table meaning true predictions out of actuals is given by
# (TruePositive + TrueNegative ) / (TP+TN+FP+FN)
accuracy_tabtraining0.4<-sum(diag(tabtraining0.4))/sum(tabtraining0.4)
accuracy_tabtraining0.4

#Verifying our Model on Testing dataset
# ------------------------------------
# checking actual 0's & 1's in testing dataset
table(testing$default10yr)

# We have below Confusion matrix as for testing
tabtesting0.4<-table(ActualValue=testing$default10yr,PredictedValue=res1>=0.4) #checked with probability cutoff = 0.4
tabtesting0.4

## Verifying our Model with Probability 0.4 for the testing dataset
#calculating Sensitivity & Specificity for dataset testing for  p=0.4
sensitivitytesting0.4<-tabtesting0.4[4]/(tabtesting0.4[2]+tabtesting0.4[4])
sensitivitytesting0.4
specificitytesting0.4<-tabtesting0.4[1]/(tabtesting0.4[1] + tabtesting0.4[3])
specificitytesting0.4

# Classification Accuracy for group 0 for testing dataset is equal to  "specificitytesting0.4" calculated as shown in output

# Classification Accuracy for group 1 is equal to  calculated as "sensitivitytesting0.4" shown in  output

# Classification Accuracy of our model for testing dataset is given by  equal to
classification_accuracy_testing0.4<- (sensitivitytesting0.4+specificitytesting0.4)/2
classification_accuracy_testing0.4

#Overall accuracy  found by using the table,meaning true total predictions out of total actuals is given by
# (TruePositive + TrueNegative ) / (TP+TN+FP+FN)
accuracy_tabtesting0.4<-sum(diag(tabtesting0.4))/sum(tabtesting0.4)
accuracy_tabtesting0.4

#Our Logistic Regression model "model" is a good model which is able to predict  the data which was even not supplied to it i.e. testing dataset. As model was trained with training dataset. We get good  Overall Accuracy and also model is less risky .

# Calculating Area under ROC curve
library(verification)
roc.area(training$default10yr,res)
# Area unde ROC curve is equal to as shown below


# Calculating Nagelkerke R square

library(fmsb)
NagelkerkeR2(model)
# our Nagelkerke R square is calculated as shownabove .

logLik(model)
# The probability of the observed results given the parameter estimates is known as the Likelihood. Since the likelihood is a small number less than 1, it is customary to use -2 times the log likelihood (-2LL) as an estimate of how well the model fits the data. A good model is one that results in a high likelihood of the observed results. This translates into a small value for -2LL (if a model fits perfectly, the likelihood=1 and -2LL=0)

#This is far from zero, however because there is no upper boundary for -2LL it is difficult to make a statement about the meaning of the score.
# It is more often used to see whether adding additional variables to the model leads to a significant reduction in the -2LL.