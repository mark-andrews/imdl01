library(tidyverse)
library(mlr) # similar to Python's sci-kit learn (sklearn)


# get data ----------------------------------------------------------------

affairs_df <- read_csv("https://raw.githubusercontent.com/mark-andrews/imdl01/master/data/affairs.csv")

training_df <- affairs_df %>%
  mutate(cheater = affairs > 0) %>% 
  mutate(across(where(is.character), as.factor)) %>% 
  select(-affairs) %>% 
  relocate(cheater) %>% 
  as.data.frame()

# data_df (all the data)
# is split to
# training_df, test_df (80/20)
# train() with training_df
# predict with test_df

# Make our "task" and our "learner" ---------------------------------------

# make the "task", which is a classification problem
affairsTask <- makeClassifTask(data = training_df, 
                               target = 'cheater')

# make the "learner", which is a classifier method, namely logistic regression
logReg <- makeLearner("classif.logreg", predict.type = 'prob')

# apply the "learner" to the "task", aka train it, aka fit it 
logRegTrained <- train(logReg, affairsTask)


# evaluate predictive performance -----------------------------------------

p <- predict(logRegTrained, newdata = training_df)
calculateConfusionMatrix(p)
calculateROCMeasures(p)

# Get model ---------------------------------------------------------------

M <- getLearnerModel(logRegTrained)


# Cross validation etc ----------------------------------------------------

kfold_cv <- makeResampleDesc(method = 'RepCV', 
                             folds = 10, 
                             reps = 10, 
                             stratify = TRUE)

logRegKfold <- resample(logReg, 
                        affairsTask, 
                        resampling = kfold_cv,
                        measures = list(acc, mmce, ppv, tpr, fpr, fdr, f1))

logRegKfold$aggr


# ROC and AUC -------------------------------------------------------------

roc_df <- generateThreshVsPerfData(p, 
                                   measures = list(fpr, tpr))
plotROCCurves(roc_df)
performance(p, measures = auc)


# Digit classification using logistic classifier --------------------------

mnist_df <- readRDS('data/mnist.Rds')


# plot digits function ----------------------------------------------------


plot_mnist <- function(data_df, rm_label = F){
  plt <- data_df %>% pivot_longer(cols = starts_with('px__'),
                                  names_to = c('x', 'y'),
                                  names_pattern = 'px__([0-9]*)_([0-9]*)',
                                  values_to = 'value') %>% 
    mutate(across(c(x,y), as.numeric)) %>% 
    ggplot(aes(x, y, fill = value)) +
    geom_tile() +
    facet_wrap(~ instance + label) +
    scale_fill_gradient(low = 'black', high = 'white') 
  
  if (rm_label){
    plt + theme(
      strip.background = element_blank(),
      strip.text.x = element_blank()
    )
  } else {
    plt
  }
  
}


plot_mnist(mnist_df %>% sample_n(16))



# Subsample the data frame to the 3 and 7 ---------------------------------

mnist_df2 <- mnist_df %>%
    filter(label %in% c(3, 7))

X <- mnist_df2 %>% select(starts_with('px__')) %>% as.matrix()
nzv <- caret::nearZeroVar(X)
image(matrix(1:(28^2) %in% nzv, 28, 28))

col_index <- setdiff(1:ncol(X), nzv)

X <- X[, col_index]
mu <- mean(X)
s <- sd(as.vector(X))
data_df <- as.data.frame((X - mu)/s) %>% 
  mutate(target = if_else(mnist_df2$label == 7, '7', '3'))

# Test/train split --------------------------------------------------------

train_idx <- sample(1:nrow(data_df), 0.8 * nrow(data_df))
test_idx <- setdiff(1:nrow(data_df), train_idx)

train_df <- data_df[train_idx,]
test_df <- data_df[test_idx,]

# Make Task and Learner ---------------------------------------------------

mnistTask <- makeClassifTask(data = train_df, target = 'target')
mnistLogReg <- makeLearner('classif.logreg', predict.type = 'prob')
mnistLogRegTrained <- train(mnistLogReg, mnistTask)


# predictions -------------------------------------------------------------

p <- predict(mnistLogRegTrained, newdata = test_df)
calculateConfusionMatrix(p)
calculateROCMeasures(p)


# Naive bayes classifier --------------------------------------------------

library(mlbench)
data("HouseVotes84")


# Make the Task and the learner -------------------------------------------

votesTask <- makeClassifTask(data = HouseVotes84, target = 'Class')
nbayesLearner <- makeLearner('classif.naiveBayes')
nbayesLearnerTrained <- train(nbayesLearner, votesTask)

# predictions -------------------------------------------------------------

p <- predict(nbayesLearnerTrained, newdata = HouseVotes84)
calculateROCMeasures(p)

# Cross-validation --------------------------------------------------------

kfold_cv <- makeResampleDesc(method = 'RepCV',
                             folds = 20,
                             reps = 10,
                             stratify = T)

nbayesCV <- resample(learner = nbayesLearner,
                     task = votesTask,
                     resampling = kfold_cv,
                     measures = list(acc, mmce, fpr, tpr, ppv, f1))

nbayesCV$aggr

# support vector machines -------------------------------------------------

library(kernlab)
data(spam)

table(spam$type)


# Make task and learner ---------------------------------------------------

spamTask <- makeClassifTask(data = spam, target = 'type')
svm_learner <- makeLearner('classif.svm', kernel = 'linear')
svm_trained <- train(svm_learner, spamTask)


# performance test --------------------------------------------------------

p <- predict(svm_trained, newdata = spam)
calculateConfusionMatrix(p)
calculateROCMeasures(p)

# Cross validation --------------------------------------------------------

kfold_cv <- makeResampleDesc(method = 'RepCV', 
                             folds = 10, reps = 3, 
                             stratify = T)

spam_svm_cv <- resample(learner = svm_learner, 
                        task = spamTask,
                        resampling = kfold_cv, 
                        measures = list(ppv, tpr, f1, fpr, fdr, acc))

spam_svm_cv$aggr


# optimize hyper-parameters -----------------------------------------------

getParamSet(svm_learner)

svm_param_space <- makeParamSet(
  makeNumericParam('cost', lower = 0.1, upper = 10),
  makeNumericParam('gamma', lower = 0.1, upper = 10)
)

randSearch <- makeTuneControlRandom(maxit = 25)

cv_for_tuning <- makeResampleDesc('Holdout', split = 2/3)

svm_tuned <- tuneParams('classif.svm',
                        task = spamTask,
                        resampling = cv_for_tuning,
                        par.set = svm_param_space,
                        control = randSearch)

svm_tuned$x

# set these as the values of the hyperparameters
spam_svm_tuned <- setHyperPars(makeLearner('classif.svm'), 
                               par.vals = svm_tuned$x)

# train with this tuned up learner
spam_svm_tuned_trained <- train(spam_svm_tuned, spamTask)

# performance check
p <- predict(spam_svm_tuned_trained, newdata = spam)
calculateROCMeasures(p)
calculateConfusionMatrix(p)

# for comparison
p_linear <- predict(svm_trained, newdata = spam)
calculateROCMeasures(p_linear)

calculateConfusionMatrix(p_linear)


# CV for tuned ------------------------------------------------------------

kfold_cv <- makeResampleDesc('CV')
spam_svm_tuned_cv <- resample(learner = spam_svm_tuned,
                              task = spamTask,
                              resampling = kfold_cv,
                              measures = list(ppv, tpr, f1, fpr, fdr, acc))
spam_svm_tuned_cv$aggr
