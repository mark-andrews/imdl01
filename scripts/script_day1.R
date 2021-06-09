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
