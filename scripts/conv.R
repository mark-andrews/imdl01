library(torch)
library(torchvision)

# Download mnist (handwritten digits) training and test sets
# these will download to your working directory
train_ds <- mnist_dataset(
  ".",
  download = TRUE,
  train = TRUE,
  transform = transform_to_tensor
)

test_ds <- mnist_dataset(
  ".",
  download = TRUE,
  train = FALSE,
  transform = transform_to_tensor
)


# but the data into batches of size 32 (32 images)
# this is for computational efficiency
train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE)
test_dl <- dataloader(test_ds, batch_size = 32)

# Make the conv net -------------------------------------------------------

net <- nn_module(
  
  "mnist_convnet",
  
  # create the various layers and functions
  # where do numbers like 9216 come from?
  # Play with functions like in `forward`
  # to see how the tesors are transformed in size by these operations
  initialize = function() {
    self$conv1 <- nn_conv2d(1, 32, 3)
    self$conv2 <- nn_conv2d(32, 64, 3)
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, 10)
  },
  
  forward = function(x) {
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      self$conv2() %>%
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      self$dropout1() %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$dropout2() %>%
      self$fc2()
  }
)

model <- net() # create an instance of the above conv net
optimizer <- optim_adam(model$parameters)

# Train it (this will take time ~ 20-30 mins on a high end cpu)
for (epoch in 1:5) {
  
  l <- c()
  
  for (b in enumerate(train_dl)) {
    # make sure each batch's gradient updates are calculated from a fresh start
    optimizer$zero_grad()
    # get model predictions
    output <- model(b[[1]])
    # calculate loss
    loss <- nnf_cross_entropy(output, b[[2]])
    # calculate gradient
    loss$backward()
    # apply weight updates
    optimizer$step()
    # track losses
    l <- c(l, loss$item())
  }
  
  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}
