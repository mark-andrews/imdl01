library(tidyverse)
library(mlr)
library(mlbench)

data(Zoo)
zoo_df <- as_tibble(Zoo)

Zoo <- mutate(Zoo,
              across(where(is.logical), as.factor)
)


# Make our Task and our Learner -------------------------------------------

ZooTask <- makeClassifTask(data = Zoo, target = 'type')
dTree <- makeLearner('classif.rpart')
dTreeTrained <- train(dTree, ZooTask)


# Visualize the tree ------------------------------------------------------

library(rpart.plot)
dtree_model <- getLearnerModel(dTreeTrained)
rpart.plot(dtree_model, type = 5)
dtree_model


# performance check -------------------------------------------------------

p <- predict(dTreeTrained, newdata = Zoo)
calculateConfusionMatrix(p)


# Cross validation --------------------------------------------------------

kfold_cv <- makeResampleDesc(method = 'RepCV', folds = 10, reps = 10)

dTreeCV <- resample(learner = dTree,
                    task = ZooTask,
                    resampling = kfold_cv,
                    measures = list(acc, mmce))

dTreeCV$aggr


# Optimize the tunable parameters -----------------------------------------

getParamSet(dTree)

dTreeParamSpace <- makeParamSet(
  makeIntegerParam('minsplit', lower = 3, upper = 25),
  makeIntegerParam('minbucket', lower = 3, upper = 10),
  makeNumericParam('cp', lower = 0.005, upper = 0.1),
  makeIntegerParam('maxdepth', lower = 3, upper = 10)
)

randSearch <- makeTuneControlRandom(maxit = 250)

cvForTuning <- makeResampleDesc('CV')

tunedDtreeParams <- tuneParams(learner = dTree,
                               task = ZooTask,
                               resampling = cvForTuning,
                               par.set = dTreeParamSpace,
                               control  = randSearch)

tunedDtreeParams$x

# set these optimal values
dtree_tuned <- setHyperPars(dTree, par.vals = tunedDtreeParams$x)

dtree_tuned_cv <- resample(learner = dtree_tuned,
                           task = ZooTask,
                           resampling = kfold_cv,
                           measures = list(acc, mmce))

dtree_tuned_cv$aggr                           

# viz the tree
dTreeTunedTrained <- train(dtree_tuned, ZooTask)
rpart.plot(getLearnerModel(dTreeTunedTrained),
           type = 5)

# Bootstrapping -----------------------------------------------------------

x <- rnorm(100) # 100 random numbers

bootstrapped_sample <- sample(x, size = length(x), replace = T)



# Random forests ----------------------------------------------------------

# make a random forest learner
rforest <- makeLearner('classif.randomForest')

kfold_cv <- makeResampleDesc(method = 'RepCV', 
                             folds = 10, 
                             reps = 10)
rforest_cv <- resample(learner = rforest,
                       task = ZooTask,
                       resampling = kfold_cv,
                       measures = list(acc, mmce))

rforest_cv$aggr

# optimize ----------------------------------------------------------------

# needs some work

# rforestParamSpace <- makeParamSet(
#   makeIntegerParam('ntree', lower = 250, upper = 250),
#   makeIntegerParam('nodesize', lower = 1, upper = 5),
#   makeIntegerParam('maxnodes', lower = 1, upper = 20)
# )
# 
# randSearch <- makeTuneControlRandom(maxit = 100)
# cvForTuning_rf <- makeResampleDesc('CV', iters = 5)
# 
# rforest_tuning <- tuneParams(learner = rforest,
#                              task = ZooTask,
#                              par.set = rforestParamSpace,
#                              control = randSearch,
#                              resampling = cvForTuning_rf)
# 
# rforest_tuning$x
# 
# rforest_tuned <- setHyperPars(rforest, par.vals = rforest_tuning$x)
# rforest_tuned_cv <- resample(learner = rforest_tuned,
#                        task = ZooTask,
#                        resampling = kfold_cv,
#                        measures = list(acc, mmce))
# 
# rforest_tuned_cv$aggr




# Kmeans ------------------------------------------------------------------

blobs_df <- read_csv("https://raw.githubusercontent.com/mark-andrews/imdl01/master/data/blobs3.csv") %>% 
  mutate(label = factor(label))

ggplot(blobs_df, 
       aes(x = x, y = y, colour = label)
) + geom_point(size = 3) + ggtitle('Labelled')


# Make task and learner ---------------------------------------------------

blobsTask <- makeClusterTask(data = select(blobs_df, -label))
k_means <- makeLearner('cluster.kmeans', 
                       par.vals = list(iter.max = 250, nstart = 10))

getParamSet(k_means)

k_meansParamSet <- makeParamSet(
  makeDiscreteParam('centers', values = 1:8)
)

gridSearch <- makeTuneControlGrid()

kfold <- makeResampleDesc('RepCV', folds = 10, reps = 10)

k_means_tuning <- tuneParams(learner = k_means,
                             task = blobsTask,
                             resampling = kfold,
                             control = gridSearch,
                             par.set = k_meansParamSet)

k_means_tuning$x

k_means_tuned <- setHyperPars(k_means, 
                              par.vals = k_means_tuning$x)


k_means_tuned_trained <- train(k_means_tuned, blobsTask)

k_means_model <- getLearnerModel(k_means_tuned_trained)

k_means_model$centers
k_means_model$cluster

blobs_df %>% 
  mutate(cluster = factor(k_means_model$cluster)) %>% 
  ggplot(aes(x = x, 
             y = y, 
             colour = cluster)
  ) + geom_point(size = 3)+ ggtitle('Predicted')


# Mixture models ----------------------------------------------------------

faithful %>% 
  ggplot(aes(x = eruptions, y = waiting)) + geom_point()

faithful %>% 
  ggplot(aes(x = eruptions, y = waiting)) + geom_density_2d()

# install.packages("mclust")
library(mclust)

faithful_bic <- mclustBIC(faithful)
plot(faithful_bic)
summary(faithful_bic)

M <- Mclust(faithful, x = faithful_bic)
summary(M)
summary(M, parameters = T)
plot(M, what = 'classification')
plot(M, what = 'density')
