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

names(dataTesting)

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
dim(training)
few_bad <- bad_count <= 0.7*length(training$classe)
training <- training[,few_bad]
validation <- validation[,few_bad]
dim(training)
# This reduced the possible predictors from 160 to 60.

# Remove some of the variables that we don't want to try to use to predict the
# outcome. These include the user and timestamps (this isn't time series data).
# Also, remove the new_window variable because it has very little variation
table(training$new_window)
training <- training[,-c(1, 2, 3, 4, 5, 6)]
validation <- validation[,-c(1, 2, 3, 4, 5, 6)]

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
mbmRF <- system.time({
    
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
# modBoost <- train(classe ~., data=training, method="gbm", 
#                   shrinkage = 0.01,
#                   trControl = BoostControl, 
#                   verbose="FALSE")
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
print(cmLDA)


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
                   trControl = NBControl,
                   
                   verbose="FALSE"
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
FinalModel <- function(x) {
    predRFValid <- predict(modRF, x)
    cmRFValid <- confusionMatrix(predRFValid, validation$classe)$overall
    
    predBoostValid <- predict(modBoost, x)
    cmBoostValid <- confusionMatrix(predBoostValid, validation$classe)$overall
    
    predLDAValid <- predict(modLDA, x)
    cmLDAValid <- confusionMatrix(predLDAValid, validation$classe)$overall
    
    predNBValid <- predict(modNB, x)
    cmNBValid <- confusionMatrix(predNBValid, validation$classe)$overall
    
    mbmCombValid <- system.time({
        dataCombValid <- data.frame(predRFValid,
                                    predBoostValid, 
                                    predLDAValid, 
                                    predNBValid, 
                                    classe = x$classe)
        CombControl <- trainControl(method="cv", number=5, allowParallel = TRUE)
        modRFCombValid <- train(classe ~., data=dataCombValid, 
                                method="rf", 
                                trControl = CombControl, 
                                verbose="FALSE")
    })
    predCombValid <- predict(modRFCombValid, x)
    cmRFCombValid <- confusionMatrix(predCombValid, x$classe)$overall
}
FinalModel(validation)
Results <- data.frame(RF = c(TrainingAccuracy=as.numeric(modRF$results[["Accuracy"]]), 
                             Validation=AccuracycmRFValid[["Accuracy"]],
                             TrainingTime=mbmRF[["user.self"]] + mbmRF[["sys.self"]]), 
                      Boost = c(TrainingAccuracy=as.numeric(modBoost$results[["Accuracy"]][9]), 
                                ValidationAccuracy=cmBoostValid[["Accuracy"]],
                                TrainingTime=mbmBoost[["user.self"]] + mbmBoost[["sys.self"]]), 
                      LDA = c(TrainingAccuracy=as.numeric(modLDA$results[["Accuracy"]]), 
                              ValidationAccuracy=cmLDAValid[["Accuracy"]],
                              TrainingTime=mbmLDA[["user.self"]] + mbmLDA[["sys.self"]]), 
                      NB = c(TrainingAccuracy=as.numeric(modNB$results[["Accuracy"]][2]), 
                             ValidationAccuracy=cmNBValid[["Accuracy"]],
                             TrainingTime=mbmNB[["user.self"]] + mbmNB[["sys.self"]]), 
                      Combined = c(TrainingAccuracy=as.numeric(modRFComb$results[["Accuracy"]][1]), 
                                   ValidationAccuracy=cmRFCombValid[["Accuracy"]],
                                   TrainingTime=mbmCombTrain[["user.self"]] + mbmCombTrain[["sys.self"]]))

stopCluster(cluster)
registerDoSEQ()