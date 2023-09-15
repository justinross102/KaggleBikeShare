##
## Bike Share Analysis Code
##

# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)

# Read in the Data
bikes <- vroom("./train.csv") %>% 
  select(-casual, -registered)
test_data <- vroom("./test.csv")

# cleaning ----------------------------------------------------------------


# feature engineering -----------------------------------------------------

my_recipe <- recipe(count ~ ., bikes) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # turn 4s into 3s
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_num2factor(weather, levels=c("partly_cloudy", "misty", "rainy")) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("no", "yes")))

prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, bikes)
bake(prepped_recipe, test_data)


# linear regression -------------------------------------------------------

# set up linear regression model
my_mod <- linear_reg() %>% 
  set_engine("lm")

bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod) %>% 
  fit(data = bikes)

# use fit to predict
bikes_predictions <- predict(bike_workflow,
                             new_data = test_data)

# round negative values to zero
bikes_predictions[bikes_predictions < 0] <- 0

# create dataset from test "datetime" and corresponding predictions
predictions <- data.frame(test_data$datetime, bikes_predictions)
colnames(predictions) = c("datetime", "count")
predictions$datetime <- as.character(predictions$datetime)

write_csv(predictions, "predictions.csv")






