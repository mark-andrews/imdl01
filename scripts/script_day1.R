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
