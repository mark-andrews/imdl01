library(torch)
library(tidyverse)

gvhd_df <- read_csv("https://raw.githubusercontent.com/mark-andrews/imdl01/master/data/gvhd.csv")

X_train <- gvhd_df %>% select(starts_with('CD')) %>% as.matrix()
y_train <- gvhd_df %>% pull(type)

X_train <- torch_tensor(X_train, dtype = torch_float())
y_train <- torch_tensor(y_train, dtype = torch_long())

mlp <- nn_sequential(
  
  ## Input-to-hidden layer
  nn_linear(4, 8),
  nn_relu(),
  
  # hidden-to-output layer
  nn_linear(8, 2),
  nn_softmax(2)
  
)

# test it
mlp(X_train)

criterion <- nn_cross_entropy_loss()
optimizer <- optim_adam(mlp$parameters, lr = 0.01)

iters <- 500

N <- nrow(gvhd_df) # sample size

for (i in 1:500){

  # clear the calculations of the gradients
  optimizer$zero_grad()
  
  # forward pass
  y_pred <- mlp(X_train)
  
  # calculate error
  loss <- criterion(y_pred, y_train)
  
  # backward pass
  # backpropagation of error 
  loss$backward()
  
  # make the step in parameter space 
  optimizer$step()
    
  if (i %% 1 == 0) {
    accuracy <- as_array(sum(y_train == y_pred$argmax(dim=2)) / N)
    cat('iter', i, 'accuracy', accuracy, '\n')
  }
  
}
