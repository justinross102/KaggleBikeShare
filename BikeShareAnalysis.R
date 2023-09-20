##
## Bike Share Analysis Code
##

# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(poissonreg)

# Read in the Data
bikes <- vroom("./train.csv")
test_data <- vroom("./test.csv")

# cleaning ----------------------------------------------------------------

# Remove casual and registered because we can't use them to predict
bikes <- bikes %>%
  select(-casual, - registered)

# feature engineering -----------------------------------------------------

my_recipe <- recipe(count~., data=bikes) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("sunny", "mist", "rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("spring", "summer", "fall", "winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("no", "yes"))) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime)
  
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = bikes) #Make sure recipe work on train
bake(prepped_recipe, new_data = test_data) #Make sure recipe works on test


# linear regression -------------------------------------------------------

# set up linear regression model
my_mod <- linear_reg() %>% 
  set_engine("lm")

bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod) %>% 
  fit(data = bikes)

# Look at the fitted LM model
extract_fit_engine(bike_workflow) %>%
  summary()

# Get Predictions for test set AND format for Kaggle submission
predictions <- predict(bike_workflow, new_data = test_data) %>%
  bind_cols(., test_data) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # only keep datetime and predictions
  rename(count=.pred) %>% # rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% # round negatives up to zero
  mutate(datetime=as.character(format(datetime))) # needed for Kaggle submission

# Write predictions to CSV file
vroom_write(x=predictions, file="./test_predictions.csv", delim=",")


# poisson regression ------------------------------------------------------

pois_mod <- poisson_reg() %>% 
  set_engine("glm") # generalized linear model

bike_pois_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(pois_mod) %>% 
  fit(data = bikes)

# Get Predictions for test set AND format for Kaggle submission
pois_predictions <- predict(bike_pois_workflow, new_data = test_data) %>%
  bind_cols(., test_data) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # only keep datetime and predictions
  rename(count=.pred) %>% # rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% # round negatives up to zero
  mutate(datetime=as.character(format(datetime))) # needed for Kaggle submission

vroom_write(x = pois_predictions, file="./poisson_predictions.csv", delim=",")


# penalized regression ----------------------------------------------------

log_bikes <- bikes %>%
  mutate(count=log(count))

## Create a recipe
pen_reg_recipe <- recipe(count~., data=log_bikes) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("sunny", "mist", "rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("spring", "summer", "fall", "winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("no", "yes"))) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% # make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1

prepped_pen_reg_recipe <- prep(pen_reg_recipe)
#bake(prepped_pen_reg_recipe, new_data = log_bikes) # Make sure recipe works on test
#bake(prepped_pen_reg_recipe, new_data = test_data) # Make sure recipe works on train

## Penalized regression model
preg_model <- linear_reg(penalty=0.0000000001, mixture=1) %>% # Set model and tuning
  set_engine("glmnet") 

# set workflow
preg_wf <- workflow() %>%
  add_recipe(pen_reg_recipe) %>%
  add_model(preg_model) %>%
  fit(data=log_bikes)

## Get Predictions for test set AND format for Kaggle
log_lin_preds <- predict(preg_wf, new_data = test_data) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

 # Write predictions to CSV
vroom_write(x=log_lin_preds, file="./LogLinearPreds.csv", delim=",")


