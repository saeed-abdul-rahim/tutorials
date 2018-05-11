library(keras)

model <- keras_model_sequential()

model %>%
  layer_conv_2d(32, c(3,3), input_shape = c(256, 256, 3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(32, c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(128) %>%
  layer_activation("relu") %>%
  layer_dense(128) %>%
  layer_activation("relu") %>%
  layer_dense(64) %>%
  layer_activation("relu") %>%
  layer_dense(64) %>%
  layer_activation("relu") %>%
  layer_dense(32) %>%
  layer_activation("relu") %>%
  layer_dense(2) %>%
  layer_activation("softmax")
  
opt <- optimizer_adam()
model %>%
  compile(loss = "categorical_crossentropy", optimizer = opt, metrics = "accuracy")
  
summary(model)

train_gen <- image_data_generator(rescale = 1/255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = T)
                                  
test_gen <- image_data_generator(rescale = 1/255)

model %>%
  fit_generator(flow_images_from_directory('dataset/training_set',
                                           train_gen),
                steps_per_epoch = 50, epochs = 30,
                validation_data = flow_images_from_directory('dataset/test_set',
                                                             test_gen))
