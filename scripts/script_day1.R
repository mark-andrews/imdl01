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


# Make our "task" and our "learner" ---------------------------------------

affairsTask <- makeClassifTask(data = training_df, target = 'cheater')
