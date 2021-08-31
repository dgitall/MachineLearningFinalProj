library(caret)
library(ggplot2)
library(randomForest)
library(corrplot)


set.seed(123)

# Setup parallel processing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

## Download and unzip the zip file from the course website
#     NOTE: checks to see if this was already done and skips if yes


destFolder = "./data"

if(!file.exists(destFolder)) {dir.create(destFolder)}

fileUrl <- "https:/d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
destFile = "./data/pml-training.csv"
if(!file.exists(destFile)) {
    download.file(fileUrl, destfile = destFile, method = "curl")
}

dataTraining <- read.csv(destFile)

fileUrl <- "https:/d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
destFile = "./data/pml-testing.csv"
if(!file.exists(destFile)) {
    download.file(fileUrl, destfile = destFile, method = "curl")
}

dataTesting <- read.csv(destFile)

inTrain <- createDataPartition(y = dataTraining$classe,
                               p = 0.7,
                               list = FALSE)
dataTraining$classe <- as.factor(dataTraining$classe)
training <- dataTraining[inTrain,]; validation <- dataTraining[-inTrain,]

print(table(training$classe))

# There are a lot of NAs
print(sum(is.na(training)))
# Get a count in each column of either NA, blank, or Div/0 errors
bad_count <-sapply(training, function(x) sum(is.na(x) | x == "" | x == "#DIV/0!"))
print(bad_count)
# Determine which columns are more than half NAs and remove them from the training
# and validation sets (only look at the training set but change both)

few_bad <- bad_count <= 0.7*length(training$classe)
training <- training[,few_bad]
validation <- validation[,few_bad]
dataQuiz <- dataTesting[,few_bad]

# This reduced the possible predictors from 160 to 60.

# Remove some of the variables that we don't want to try to use to predict the
# outcome. These include the user and timestamps (this isn't time series data).
# Also, remove the new_window variable because it has very little variation
table(training$new_window)
unneeded <- c(1, 2, 3, 4, 5, 6)
training <- training[,-unneeded]
validation <- validation[,-unneeded]
dataQuiz <- dataQuiz[,-unneeded]

# Look at the correlation matrix
corr <- cor(training[,-54])
corrplot(corr, type="upper", order="hclust", 
         sig.level = 0.01, insig = "blank")
# This shows that a large number of our variable have a low correlation. This is
# a good indication that we can find a good model with the variables available.

# However, we should remove the highly correlated variables to avoid problems
# in our modeling
dim(training)
removeCorr <- findCorrelation(corr, cutoff = 0.9)
print("Indices of highly correlated variables to remove")
removeCorr
training <- training[,-removeCorr]
validation <- validation[,-removeCorr]
dataQuiz <- dataQuiz[,-removeCorr]
dim(training)

# Set the remaining predictors to numeric
# training <- training %>% select(-classe) %>% mutate_all(as.numeric)
# validation <- validation %>% select(-classe) %>% mutate_all(as.numeric)

# Random Forest
# Tune the random forest
# https://rpubs.com/phamdinhkhanh/389752
print("Start RF Tune")
mbmRFtune <- system.time({
    
    RFControl <- trainControl(method='repeatedcv',
                              number=10,
                              repeats=3,
                              search='grid',
                              allowParallel = TRUE)
    # Run with various different values of mtry from 1 to 15
    tunegrid <- expand.grid(.mtry = (1:15))
    modRF <- train(classe ~., data=training,
                   method="rf",
                   ntree = 150,
                   trControl = RFControl,
                   metric = 'Accuracy',
                   tuneGrid = tunegrid,
                   verbose="FALSE")
})
plot(modRF)
# Use the mtry = 8 as picked out from the approach above where we don't get additional 
# accuracy gains
mbmRF<- system.time({
    
    RFControl <- trainControl(method='repeatedcv',
                              number=10,
                              repeats=3,
                              search='grid',
                              allowParallel = TRUE)
    # Run with mtry=8
    tunegrid <- expand.grid(.mtry = 4)
    modRF <- train(classe ~., data=training,
                   method="rf",
                   ntree = 150,
                   trControl = RFControl,
                   metric = 'Accuracy',
                   tuneGrid = tunegrid,
                   verbose="FALSE")
})

cmRF <- confusionMatrix.train(modRF)
print(cmRF)
# Stochastic Gradient Boosting
print("Start Boost")
# Tuning Boost
mbmBoost <- system.time({
    
    BoostControl <- trainControl(## 10-fold CV
        method = "repeatedcv",
        number = 10,
        ## repeated ten times
        repeats = 10,
        allowParallel = TRUE)
    modBoost <- train(classe ~., data=training, method="gbm", 
                      trControl = BoostControl, 
                      verbose="FALSE")
})
cmBoost <- confusionMatrix.train(modBoost)
print(modBoost)
print(cmBoost)
summary(modBoost)

# Factor-Based Linear Discriminant Analysis
print("Start LDA")
mbmLDA <- system.time({
    
    LDAControl <- trainControl(allowParallel = TRUE)
    modLDA <- train(classe ~., data=training, 
                    method="lda", 
                    trControl = LDAControl, 
                    verbose="FALSE")
})
cmLDA <- confusionMatrix.train(modLDA)


# Naive Bayes
print("Start NB")
# NBControl <- trainControl(allowParallel = TRUE)
mbmNB <- system.time({
    
    NBControl <- trainControl(## 10-fold CV
        method = "cv",
        number = 10,
        allowParallel = TRUE)
    modNB <- train(classe ~., data=training, 
                   method="nb",
                   trControl = NBControl
    )
})

print(modNB)
cmNB<- confusionMatrix.train(modNB)
print(cmNB)

# Combine the model results on the training set
print("Start Combined")
mbmCombTrain <- system.time({
    
    predCombTrain <- data.frame(predict(modRF, training),
                                predict(modBoost, training), 
                                predict(modLDA, training), 
                                predict(modNB, training),
                                classe = training$classe)
    CombControl <- trainControl(method="cv", number=5, allowParallel = TRUE)
    modRFComb <- train(classe ~., data=predCombTrain, 
                       method="rf", 
                       trControl = CombControl, 
                       verbose="FALSE")
})
cmRFComb <- confusionMatrix.train(modRFComb)
print(cmRFComb)


print("Starting Validation")
predRFValid <- predict(modRF, validation)
cmRFValid <- confusionMatrix(predRFValid, validation$classe)$overall

predBoostValid <- predict(modBoost, validation)
cmBoostValid <- confusionMatrix(predBoostValid, validation$classe)$overall

predLDAValid <- predict(modLDA, validation)
cmLDAValid <- confusionMatrix(predLDAValid, validation$classe)$overall

predNBValid <- predict(modNB, validation)
cmNBValid <- confusionMatrix(predNBValid, validation$classe)$overall

mbmCombValid <- system.time({
    dataCombValid <- data.frame(predRFValid,
                                predBoostValid, 
                                predLDAValid, 
                                predNBValid, 
                                classe = validation$classe)
    CombControl <- trainControl(method="cv", number=5, allowParallel = TRUE)
    modRFCombValid <- train(classe ~., data=dataCombValid, 
                            method="rf", 
                            trControl = CombControl, 
                            verbose="FALSE")
})
predCombValid <- predict(modRFCombValid, validation)
cmRFCombValid <- confusionMatrix(predCombValid, validation$classe)$overall


timeRF <- mbmRF[["user.self"]] + mbmRF[["sys.self"]] + 
    mbmRFtune[["user.self"]] + mbmRFtune[["sys.self"]]
timeBoost <- mbmBoost[["user.self"]] + mbmBoost[["sys.self"]]
timeLDA <- mbmLDA[["user.self"]] + mbmLDA[["sys.self"]]
timeNB <- mbmNB[["user.self"]] + mbmNB[["sys.self"]]
timeComb <- timeRF + timeBoost + timeLDA + timeNB + 
    mbmCombValid[["user.self"]] + mbmCombValid[["sys.self"]]
Results <- data.frame(RF = c(TrainingAccuracy=as.numeric(modRF$results[["Accuracy"]]), 
                             ValidationAccuracy=cmRFValid[["Accuracy"]],
                             TrainingTime=timeRF), 
                      Boost = c(TrainingAccuracy=as.numeric(modBoost$results[["Accuracy"]][9]), 
                                ValidationAccuracy=cmBoostValid[["Accuracy"]],
                                TrainingTime=timeBoost), 
                      LDA = c(TrainingAccuracy=as.numeric(modLDA$results[["Accuracy"]]), 
                              ValidationAccuracy=cmLDAValid[["Accuracy"]],
                              TrainingTime=timeLDA), 
                      NB = c(TrainingAccuracy=as.numeric(modNB$results[["Accuracy"]][2]), 
                             ValidationAccuracy=cmNBValid[["Accuracy"]],
                             TrainingTime=timeNB), 
                      Combined = c(TrainingAccuracy=as.numeric(modRFComb$results[["Accuracy"]][1]),
                                   ValidationAccuracy=cmRFCombValid[["Accuracy"]],
                                   TrainingTime=timeComb))

library(reactable)
reactable(Results,
          fullWidth = FALSE,
          bordered = TRUE,
          highlight = TRUE,
          outlined = TRUE,
          defaultColDef = colDef(format = colFormat(digits = 5)),
          # rownames = list("Model", "Training Accuracy", "Validation Accuracy", "Training Time (s)"),
          columns=list(.rownames = colDef(name = "Model", align = "right", minWidth=180),
                       RF = colDef(name = "Random Forest**", align = "center", minWidth=110),
                       Boost = colDef(name = "Stochastic Gradient Boosting", align = "center",
                                      minWidth=110), 
                       LDA = colDef(name = "Linear Discriminant Analysis", align = "center",
                                    minWidth=110),
                       NB = colDef(name = "Naive Bayes", align = "center", minWidth=110),
                       Combined = colDef(name = "Combined (Model Stacking)***", align = "center",
                                         minWidth=110)))


predRFQuiz<- predict(modRF, dataQuiz)
predBoostQuiz <- predict(modBoost, dataQuiz)
predLDAQuiz <- predict(modLDA, dataQuiz)
predNBQuiz<- predict(modNB, dataQuiz)
dataCombQuiz <- data.frame(predRFQuiz,
                           predBoostQuiz, 
                           predLDAQuiz, 
                           predNBQuiz)
predCombQuiz <- predict(modRFComb, dataCombQuiz)
print(predCombQuiz)

library("mlbench")
library("randomForest")
library("nnet")
library(caretEnsemble)
my_control <- trainControl(
    method = "repeatedcv",
    number = 10,
    ## repeated ten times
    repeats = 10,
    savePredictions="final",
    classProbs=TRUE,
    index=createResample(training$classe, 25),
    allowParallel = TRUE
)
model_list_big <- caretList(
    classe~., data=training,
    trControl=my_control,
    metric="Accuracy",
    methodList=c("rf", "gbm", "lda", "nb"),
    tuneList=list(
        rf1=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=4)),
        gbm=caretModelSpec(method="gbm"),
        lda=caretModelSpec(method="lda"),
        nb=caretModelSpec(method="nb")
    )
)

glm_ensemble <- caretStack(
    model_list_big,
    method="glm",
    metric="Accuracy",
    trControl=trainControl(
        method="cv",
        number=10,
        # savePredictions="final",
        # classProbs=TRUE,
    )
)
model_preds2 <- model_preds
model_preds2$ensemble <- predict(glm_ensemble, newdata=validation, type="prob")
CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
colAUC(model_preds2, validation$classe)


stopCluster(cluster)
registerDoSEQ()