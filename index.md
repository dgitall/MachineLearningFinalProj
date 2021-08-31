---
title: "Practical Machine Learning: Final Project "
author: "Darrell Gerber"
output:
  html_document: 
    keep_md: yes

---




## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement -- a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

The goal of your project is to predict the manner in which they did the
exercise. This is the "classe" variable in the training set. You may use
any of the other variables to predict with. You should create a report
describing how you built your model, how you used cross validation, what
you think the expected out of sample error is, and why you made the
choices you did. You will also use your prediction model to predict 20
different test cases.

NOTE: At the time of this submission, the groupware site referenced
above is inaccessible. Therefore, we are limited to the information
available in the data files to understand the data.

NOTE TO REVIEWERS: Some of the prediction methods used are computational expensive. To cut down on computation time and still allow for model tuning, parallel processing is enabled. For more information on using parallel processing for this project and the impacts, refer to https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md. 

Note: Github repository --> https://github.com/dgitall/MachineLearningFinalProj 
  

```r
# Setup parallel processing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(123)
```

## Loading and Exploring Data

Download the data from online storage and load it into training and
testing data sets.


```r
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
```
  
Split the dataTraining data set into a training and validation sets. 


```r
inTrain <- createDataPartition(y = dataTraining$classe,
                               p = 0.7,
                               list = FALSE)
dataTraining$classe <- as.factor(dataTraining$classe)
training <- dataTraining[inTrain,]; validation <- dataTraining[-inTrain,]
```
  
There are 160 variables in the training data set, however, some of the variables are almost entirely devoid of information due to blanks, NAs, or division by zero errors.  

```r
bad_count <-sapply(training, function(x) sum(is.na(x) | x == "" | x == "#DIV/0!"))
numWithBad <- sum(bad_count==0)
```
  
60 columns have at least one of these problem entries. Just to be sure, only the variables with more than 70% valid data will be kept.  (NOTE: determine the columns to remove by looking at the training data only but apply it to both training and validation data sets).  
  

```r
dimStart <- dim(training)[2]
few_bad <- bad_count <= 0.7*length(training$classe)
training <- training[,few_bad]
validation <- validation[,few_bad]
dimBad <- dim(training)[2]
```

```r
# Keep track of the columns removed so we can apply the same the testing data set
colRemoved <- bad_count > 0.7*length(training$classe)
```
  
This reduced the variables from 160 to 60.  

We can also remove some of the variables that we don't want to use to predict the
outcome. These include user related variables and timestamps (the selection must be user agnostic and this isn't time series data).  As illustrated below, the new_window variable has very little variation so it can be removed from modeling, too. Remove the related num_window variable as well because it isn't clearly related to the instrument measurements. 

```r
table(training$new_window)
```

```
## 
##    no   yes 
## 13463   274
```

```r
unneeded <- c(1, 2, 3, 4, 5, 6, 7)
training <- training[,-unneeded]
validation <- validation[,-unneeded]

colRemoved <- c(unneeded, colRemoved)
```
  
Correlation between variables can cause many of our models to perform poorly. Plot the correlation matrix to see if we have any highly correlated predictors. 

```r
corr <- cor(training[,-length(names(training))])
corrplot(corr, type="upper", order="hclust", 
         sig.level = 0.01, insig = "blank")
```

![](index_files/figure-html/correlation-1.png)<!-- -->
  
A large number of variables have a low correlation. However, there are a few highly correlated variables. Find those variables with a high level of correlation to other variables in the data set and remove them. A cutoff of 0.9 will only remove those with the most extreme correlation. 

```r
dimb4Corr <- dim(training)[2]
removeCorr <- findCorrelation(corr, cutoff = 0.9)
print("Indices of highly correlated variables to remove")
```

```
## [1] "Indices of highly correlated variables to remove"
```

```r
removeCorr
```

```
## [1] 10  1  9  8 31 33 18
```

```r
training <- training[,-removeCorr]
validation <- validation[,-removeCorr]

colRemoved <- c(removeCorr, colRemoved)
dimAfterCorr <- dim(training)[2]
```
  
This removes another 7 variables, leaving a final data set of 45 variables we will use to predict the variable 'classe'.  

# Modeling
The general approach is to individually evaluate several unrelated modeling methods and then combine their results using model stacking. The 'classe' variable is a factor with 5 possible values, therefore, we will select prediction models appropriate for classification problems. Random Forest and Generalized Boosting models, both tree-based models, are used along with Linear Discrete Analysis and Naive Bayes, both model-based methods. There are many, many modeling options so this is only a small sample of options.   

## Random Forest
Random Forest is an extension of bagging for classification and regression trees. It should prove highly accurate but is computationally intensive and susceptible to overfitting. For that reason, we will tune the algorithm to avoid overfitting and improve performance.  

The total number of trees is reduced to 150 from the default 500. The model is then trained using values of mtry ranging from 1 to 15. (refer to https://rpubs.com/phamdinhkhanh/389752 for this and other tuning methods.)

```r
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
```

```r
plot(modRF)
```

![](index_files/figure-html/plotRFtune-1.png)<!-- -->
  
The model accuracy increases as mtry increases, however, after mtry=5 the marginal improvement in accuracy drops off significantly. Selecting mtry=5 will continue to provide high accuracy while reducing execution time and reducing the likelihood of over fitting.


```r
mbmRF <- system.time({

RFControl <- trainControl(method='repeatedcv',
                        number=10,
                        repeats=3,
                        search='grid',
                        allowParallel = TRUE)
# Run with mtry=8
tunegrid <- expand.grid(.mtry = 5)
modRF <- train(classe ~., data=training,
               method="rf",
               ntree = 150,
               trControl = RFControl,
               metric = 'Accuracy',
               tuneGrid = tunegrid,
               verbose="FALSE")
})
```
  
The final random forest model achieves a high accuracy of 0.9930357 on the training data set.  

## Stochastic Gradient Boosting Tree Model
The GBM algorithm will assemble a prediction tree while boosting weak performing predictors at each level to force better fits. Like Random Forest model, GBM models are prone to over fitting. To minimize this problem, run an autotuning version using repeated cross validation in caret. (for more information on tuning GBM, refer to https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab and  https://www.listendata.com/2015/07/gbm-boosted-models-tuning-parameters.html)  

```r
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
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2152
##      2        1.4765             nan     0.1000    0.1494
##      3        1.3846             nan     0.1000    0.1128
##      4        1.3153             nan     0.1000    0.0930
##      5        1.2563             nan     0.1000    0.0939
##      6        1.2001             nan     0.1000    0.0712
##      7        1.1556             nan     0.1000    0.0659
##      8        1.1147             nan     0.1000    0.0624
##      9        1.0769             nan     0.1000    0.0485
##     10        1.0468             nan     0.1000    0.0618
##     20        0.8051             nan     0.1000    0.0309
##     40        0.5532             nan     0.1000    0.0118
##     60        0.4271             nan     0.1000    0.0063
##     80        0.3432             nan     0.1000    0.0052
##    100        0.2828             nan     0.1000    0.0025
##    120        0.2391             nan     0.1000    0.0021
##    140        0.2050             nan     0.1000    0.0025
##    150        0.1913             nan     0.1000    0.0024
```

```r
print(modBoost)
```

```
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 10 times) 
## Summary of sample sizes: 12363, 12363, 12363, 12365, 12364, 12364, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7403370  0.6709725
##   1                  100      0.8131394  0.7636023
##   1                  150      0.8462623  0.8054677
##   2                   50      0.8524138  0.8130557
##   2                  100      0.9036400  0.8780500
##   2                  150      0.9278225  0.9086646
##   3                   50      0.8917308  0.8629011
##   3                  100      0.9385815  0.9222773
##   3                  150      0.9583748  0.9473382
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```
  
The optimization held shrinkage and n.minobsinnode constant and found the best model using n.trees = 150 and interaction.depth = 3. The final GBM model is also highly accurate at 0.9583748.  


```r
summary(modBoost)
```

![](index_files/figure-html/boostsummary-1.png)<!-- -->

```
##                                       var     rel.inf
## yaw_belt                         yaw_belt 13.24206665
## pitch_forearm               pitch_forearm  9.89196891
## magnet_dumbbell_z       magnet_dumbbell_z  7.93769637
## magnet_belt_y               magnet_belt_y  6.08988778
## gyros_belt_z                 gyros_belt_z  5.54076974
## pitch_belt                     pitch_belt  5.46265845
## roll_forearm                 roll_forearm  5.34587465
## magnet_dumbbell_y       magnet_dumbbell_y  4.90828509
## magnet_belt_z               magnet_belt_z  4.84001987
## roll_dumbbell               roll_dumbbell  3.14423131
## accel_forearm_x           accel_forearm_x  2.43185613
## total_accel_belt         total_accel_belt  2.31983956
## magnet_forearm_z         magnet_forearm_z  2.24474124
## accel_forearm_z           accel_forearm_z  2.13915761
## accel_dumbbell_y         accel_dumbbell_y  2.03499211
## magnet_dumbbell_x       magnet_dumbbell_x  1.83546965
## gyros_dumbbell_y         gyros_dumbbell_y  1.82697130
## yaw_arm                           yaw_arm  1.70726241
## accel_dumbbell_x         accel_dumbbell_x  1.63633946
## magnet_arm_z                 magnet_arm_z  1.41755854
## magnet_belt_x               magnet_belt_x  1.20786266
## accel_dumbbell_z         accel_dumbbell_z  1.16070762
## roll_arm                         roll_arm  1.09764482
## gyros_arm_y                   gyros_arm_y  1.06145926
## gyros_belt_y                 gyros_belt_y  1.04714447
## magnet_arm_y                 magnet_arm_y  0.95002020
## magnet_arm_x                 magnet_arm_x  0.85152091
## total_accel_dumbbell total_accel_dumbbell  0.84719829
## total_accel_forearm   total_accel_forearm  0.68485622
## magnet_forearm_x         magnet_forearm_x  0.66301361
## gyros_belt_x                 gyros_belt_x  0.63940238
## accel_arm_x                   accel_arm_x  0.56174246
## pitch_dumbbell             pitch_dumbbell  0.41582712
## accel_arm_y                   accel_arm_y  0.35563668
## accel_arm_z                   accel_arm_z  0.35064256
## magnet_forearm_y         magnet_forearm_y  0.33799541
## accel_forearm_y           accel_forearm_y  0.30384940
## yaw_dumbbell                 yaw_dumbbell  0.29137027
## total_accel_arm           total_accel_arm  0.23731537
## gyros_forearm_z           gyros_forearm_z  0.23317518
## gyros_forearm_y           gyros_forearm_y  0.18605266
## gyros_forearm_x           gyros_forearm_x  0.16482799
## pitch_arm                       pitch_arm  0.16012098
## yaw_forearm                   yaw_forearm  0.15185248
## gyros_arm_z                   gyros_arm_z  0.04111416
```
   
NOTE: the GBM model allows checking which variables have the highest influence on the model. The graph above shows that the top three most influential variables carry significantly more weight than the others. The top variable is yaw_belt, followed by pitch_forearm, and magnet_dumbbell_z.      
  
## Factor-Based Linear Discriminant Analysis
LDA is a fast model-based approach that assumes a multivariate Gaussian distribution in the predictors and that they have the same covariances. Effectively, it creates a model that is a series of straight lines drawn through the data. This will be a fast model to train, but it could be highly inaccurate if the data does not meet the assumptions.   


```r
mbmLDA <- system.time({

LDAControl <- trainControl(allowParallel = TRUE)
modLDA <- train(classe ~., data=training, 
                method="lda", 
                trControl = LDAControl, 
                verbose="FALSE")
})
```
  
As expected, with an accuracy on the training data of 0.6754873, the LDA method achieves far lower accuracy than either random forest or boosting.  
  
# Naive Bayes
Naive Bayes is a model-based approach like LDA, however, it assumes that all of the predictors are independent. This method should give adequate results since the highly correlated variables were removed. It should have a lower computational cost than GBM and Random Forest.

```r
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
```

```r
print(modNB)
```

```
## Naive Bayes 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12364, 12364, 12364, 12364, 12362, 12362, ... 
## Resampling results across tuning parameters:
## 
##   usekernel  Accuracy   Kappa    
##   FALSE      0.5378118  0.4242194
##    TRUE      0.7549651  0.6902744
## 
## Tuning parameter 'fL' was held constant at a value of 0
## Tuning
##  parameter 'adjust' was held constant at a value of 1
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were fL = 0, usekernel = TRUE and adjust
##  = 1.
```
The Naive Bayes model outperformed LDA with an accuracy of 0.7549651 on the training set. The model is tuned using cross validation and found the most accurate approach was to set useKernal=TRUE.  
  
## Combining models
For the final step, combine all of the models using model stacking and evaluate the accuracy on the training data set. 

```r
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
```
The accuracy of the combined results (0.9988352) was better than any of the individual models. However, the gains are marginal since the accuracy of RF and GBM were already very high. 

# Validation
Each of the models is applied to the validation data set. The accuracy of each is compared individually and the results are combined using model stacking Calculate the out-of-sample error for each method, too.   


```r
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
```

```r
timeRF <- mbmRF[["user.self"]] + mbmRF[["sys.self"]] + 
  mbmRFtune[["user.self"]] + mbmRFtune[["sys.self"]]
timeBoost <- mbmBoost[["user.self"]] + mbmBoost[["sys.self"]]
timeLDA <- mbmLDA[["user.self"]] + mbmLDA[["sys.self"]]
timeNB <- mbmNB[["user.self"]] + mbmNB[["sys.self"]]
timeComb <- timeRF + timeBoost + timeLDA + timeNB + 
  mbmCombValid[["user.self"]] + mbmCombValid[["sys.self"]]
Results <- data.frame(RF = c(TrainingAccuracy=as.numeric(modRF$results[["Accuracy"]]), 
                             ValidationAccuracy=cmRFValid[["Accuracy"]],
                             OutOfSampleError = 1-cmRFValid[["Accuracy"]],
                             TrainingTime=timeRF), 
                      Boost = c(TrainingAccuracy=as.numeric(modBoost$results[["Accuracy"]][9]), 
                                ValidationAccuracy=cmBoostValid[["Accuracy"]],
                             OutOfSampleError = 1-cmBoostValid[["Accuracy"]],
                                TrainingTime=timeBoost), 
                      LDA = c(TrainingAccuracy=as.numeric(modLDA$results[["Accuracy"]]), 
                              ValidationAccuracy=cmLDAValid[["Accuracy"]],
                             OutOfSampleError = 1-cmLDAValid[["Accuracy"]],
                              TrainingTime=timeLDA), 
                      NB = c(TrainingAccuracy=as.numeric(modNB$results[["Accuracy"]][2]), 
                             ValidationAccuracy=cmNBValid[["Accuracy"]],
                             OutOfSampleError = 1-cmNBValid[["Accuracy"]],
                             TrainingTime=timeNB), 
                      Combined = c(TrainingAccuracy=as.numeric(modRFComb$results[["Accuracy"]][1]),
                                   ValidationAccuracy=cmRFCombValid[["Accuracy"]],
                             OutOfSampleError = 1-cmRFCombValid[["Accuracy"]],
                                   TrainingTime=timeComb))

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
```

```{=html}
<div id="htmlwidget-0f067ded6adcb52db107" class="reactable html-widget" style="width:auto;height:auto;"></div>
<script type="application/json" data-for="htmlwidget-0f067ded6adcb52db107">{"x":{"tag":{"name":"Reactable","attribs":{"data":{".rownames":["TrainingAccuracy","ValidationAccuracy","OutOfSampleError","TrainingTime"],"RF":[0.993035662056434,0.994392523364486,0.00560747663551397,27.52],"Boost":[0.958374805464786,0.960407816482583,0.0395921835174171,29.97],"LDA":[0.675487276758917,0.67357689039932,0.32642310960068,1.52],"NB":[0.754965122766406,0.754630416312659,0.245369583687341,1.97000000000001],"Combined":[0.998835172311835,0.994562446898896,0.0054375531011045,64.56]},"columns":[{"accessor":".rownames","name":"Model","type":"character","format":{"cell":{"digits":5},"aggregated":{"digits":5}},"sortable":false,"filterable":false,"minWidth":180,"align":"right"},{"accessor":"RF","name":"Random Forest**","type":"numeric","format":{"cell":{"digits":5},"aggregated":{"digits":5}},"minWidth":110,"align":"center"},{"accessor":"Boost","name":"Stochastic Gradient Boosting","type":"numeric","format":{"cell":{"digits":5},"aggregated":{"digits":5}},"minWidth":110,"align":"center"},{"accessor":"LDA","name":"Linear Discriminant Analysis","type":"numeric","format":{"cell":{"digits":5},"aggregated":{"digits":5}},"minWidth":110,"align":"center"},{"accessor":"NB","name":"Naive Bayes","type":"numeric","format":{"cell":{"digits":5},"aggregated":{"digits":5}},"minWidth":110,"align":"center"},{"accessor":"Combined","name":"Combined (Model Stacking)***","type":"numeric","format":{"cell":{"digits":5},"aggregated":{"digits":5}},"minWidth":110,"align":"center"}],"defaultPageSize":10,"paginationType":"numbers","showPageInfo":true,"minRows":1,"highlight":true,"outlined":true,"bordered":true,"inline":true,"dataKey":"b550cce672db0640925955fa90bda25a","key":"b550cce672db0640925955fa90bda25a"},"children":[]},"class":"reactR_markup"},"evals":[],"jsHooks":[]}</script>
```
All of the models performed similarly well on the validation set and some even outperformed the results using training data. The random forest and GBM models performed well on the validation data indicating that concerns about over fitting are likely unfounded. The combined model stacking output continued to perform well on the validation set.

A comparison of the relative time to complete model training indicates a strong inverse relationship between computational effort and model accuracy. The Naive Bayes method, however, performed well given the significantly lower computational effort than random forest and GBM. Additionally, the random forest and GBM both had very high accuracy meaning that the marginal accuracy gain from combining the results was small. However, combining the model-based predictors with the tree-based predictors may help further reduce the impact of over fitting when applied to new real-world data.  
  
NOTEs: * All of the timing data, in seconds, is a sum of the user time and system time. It is not the apparent time of execution. These values are highly platform dependent. They should only be used for comparison between prediction methods. ** The training time for random forest includes both the time to tune the model and the final modeling run. *** The Combined (Model Stacking) time is the time to combine the validation results plus the sum of the training time for all of the constituent models.
  

```r
stopCluster(cluster)
registerDoSEQ()
```
