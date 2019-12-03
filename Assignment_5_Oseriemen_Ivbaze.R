library('ggplot2')
library(gridExtra)
library('maps')
library('dplyr')
library('caret')
library('gbm')
library('RANN')

data(scat)
#3. Check if any values are null. If there are, impute missing values using KNN. (10 points)
sum(is.na(scat))

preProcValues <- preProcess(scat, method = c("knnImpute","center","scale"))

scat_processed1 <- predict(preProcValues, scat)
scat_processed <- predict(preProcValues, scat)
sum(is.na(scat_processed))

#1. Set the Species column as the target/outcome and convert it to numeric. (5 points)
scat_processed$Species<-ifelse(scat_processed$Species=='bobcat',1, ifelse(scat_processed$Species=='coyote', 2,ifelse(scat_processed$Species=='gray_fox', 3, 0)))


#2. Remove the Month, Year, Site, Location features. (5 points)
select (scat_processed,-c(Month, Year, Site, Location))
select (scat_processed1,-c(Month, Year, Site, Location))

#4. 
#We don't need to because the categorical variables are already numerical

#With a seed of 100, 75% training, 25% testing. Build the following models: randomforest, neural
#net, naive bayes and GBM.
#a. For these models display a)model summarization and b) plot variable of importance, for
#the predictions (use the prediction set) display c) confusion matrix (60 points)
#Spliting training set into two parts based on outcome: 75% and 25%

set.seed(100)
index <- createDataPartition(scat_processed$Species, p=0.75, list=FALSE)
trainSet <- scat_processed[ index,]
testSet <- scat_processed[-index,]

set.seed(100)
index1 <- createDataPartition(scat_processed1$Species, p=0.75, list=FALSE)
trainSet1 <- scat_processed1[ index1,]
testSet1 <- scat_processed1[-index1,]


#Checking the structure of trainSet
str(trainSet)

#Feature selection using rfe in caret
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Scat_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],rfeControl = control)
Scat_Pred_Profile

predictors<-c("CN", "Mass", "segmented", "d13C", "Site")

#Do another special predictor set for naive bayes
predictors1<-names(trainSet1)[!names(trainSet1) %in% outcomeName]
Scat_Pred_Profile1 <- rfe(trainSet1[,predictors1], trainSet1[,outcomeName],rfeControl = control)
Scat_Pred_Profile1
predictors1<- c("CN", "d13C", "d15N", "Mass")

######## Training Models Using Caret ############

# For example, to apply, GBM, Random forest, Neural net:
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf', importance=T)
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet', importance=T)
model_nbayes<-train(trainSet1[,predictors1],trainSet1[,outcomeName],method='nb')

############ Variable importance estimation using caret ##################
print(model_gbm)
#Variable Importance
varImp(object=model_gbm)
#Plotting Varianle importance for GBM
gbm_plot <- plot(varImp(object=model_gbm),main="GBM - Variable Importance")
plot(varImp(object=model_gbm),main="GBM - Variable Importance")
#conufsion matrix
predictions_gbm<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
#Confusion Matrix and Statistics
confusionMatrix(table(factor(predictions_gbm, levels=1:27),factor(testSet$Species, levels=1:27)))
confusionMatrix(table(factor(predictions_gbm, levels=1:27),factor(testSet$Species, levels=1:27)))

print(model_rf)
varImp(object=model_rf)
#Plotting Varianle importance for Random Forest
rf_plot <- plot(varImp(object=model_rf),main="RF - Variable Importance")
plot(varImp(object=model_rf),main="RF - Variable Importance")
#conufsion matrix
predictions_rf<-predict.train(object=model_rf,testSet[,predictors],type="raw")
#Confusion Matrix and Statistics
confusionMatrix(table(factor(predictions_rf, levels=1:27),factor(testSet$Species, levels=1:27)))

print(model_nnet)
varImp(object=model_nnet)
#Plotting Variable importance for Neural Network
nnet_plot <- plot(varImp(object=model_nnet),main="NNET - Variable Importance")
plot(varImp(object=model_nnet),main="NNET - Variable Importance")
#GBM conufsion matrix
predictions_nnet<-predict.train(object=model_nnet,testSet[,predictors],type="raw")
#Confusion Matrix and Statistics
confusionMatrix(table(factor(predictions_nnet, levels=1:27),factor(testSet$Species, levels=1:27)))

print(model_nbayes)
varImp(object=model_nbayes)
#Plotting Variable importance for Naive Bayes
nbayes_plot <- plot(varImp(object=model_nbayes),main="NBayes - Variable Importance")
plot(varImp(object=model_nbayes),main="NBayes - Variable Importance")
#conufsion matrix
predictions_nbayes<-predict.train(object=model_nbayes,testSet1[,predictors1],type="raw")
#Confusion Matrix and Statistics
confusionMatrix(table(factor(predictions_nbayes, levels=1:27),factor(testSet$Species, levels=1:27)))

#6
nnet_cm <- confusionMatrix(table(factor(predictions_nnet, levels=1:27),factor(testSet$Species, levels=1:27)))
nnet_cm
cm_df <- data.frame("Experiment Name" = "NNET", "Accuracy" = c(nnet_cm$overall['Accuracy']), "Kappa" = c(nnet_cm$overall['Kappa']))
cm_df
cm_df[order("Accuracy")]

#7.
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
### Using tuneGrid ####
modelLookup(model='gbm')
# training the model
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)
print(model_gbm)
plot(model_gbm)

#8
grid.arrange(gbm_plot, rf_plot,nnet_plot, nbayes_plot, nrow = 2)

# 9. Which model performs the best? and why do you think this is the case? Can we accurately
#predict species on this dataset?

#The nnet model performed the best simply because it was the only model that produced any tangible results.
#It does not seem to be accurate because of the other failed results from the other models.
# Also the nnet model only has an accuracy of .48, so it is still below 50% accuracy.



                                    