##
## Bike Share Analysis Code
##

# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(poissonreg)
library(rpart)
library(ranger)
library(stacks)

# Read in the Data
train <- vroom("./train.csv")
test <- vroom("./test.csv")

# cleaning ----------------------------------------------------------------

# Remove casual and registered because we can't use them to predict
train <- train %>%
  select(-casual, - registered)

# Function for predicting and formatting data for Kaggle Submission
predict_and_format <- function(Workflow, New_data, Output_file) {
  result <- predict(Workflow, New_data) %>%
    mutate(.pred = exp(.pred)) %>%
    bind_cols(., New_data) %>%
    select(datetime, .pred) %>%
    rename(count = .pred) %>%
    mutate(datetime = as.character(format(datetime)))
  
  vroom::vroom_write(result, file = Output_file, delim = ",")
}

# feature engineering -----------------------------------------------------

log_train <- train %>%
  mutate(count=log(count))

my_recipe <- recipe(count ~ ., data = log_train) %>%
  step_mutate(weather=ifelse(weather == 4, 3, weather)) %>% # Relabel weather 4 to 3
  step_num2factor(weather, levels = c("sun", "mist", "rain")) %>% 
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% 
  step_mutate(holiday = factor(holiday, levels = c(0,1), labels = c("no", "yes"))) %>%
  step_mutate(workingday = factor(workingday,levels = c(0,1), labels = c("no", "yes"))) %>%
  step_time(datetime, features="hour") %>% # pull out individual variables from datetime
  step_date(datetime, features="dow") %>% 
  step_date(datetime, features="month") %>% 
  step_date(datetime, features="year") %>% 
  step_rm(datetime) %>% # don't need it anymore
  step_dummy(all_nominal_predictors()) %>% # make dummy variables
  step_normalize(all_numeric_predictors()) %>%  # Make mean 0, sd=1
  step_nzv(all_numeric_predictors())

prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = log_train) #Make sure recipe work on train
bake(prepped_recipe, new_data = test) #Make sure recipe works on test

# linear regression -------------------------------------------------------

# set up linear regression model
lin_mod <- linear_reg() %>% 
  set_engine("lm")

linear_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(lin_mod) %>% 
  fit(data = log_train)

# Look at the fitted LM model
extract_fit_engine(linear_workflow) %>%
  summary()

# Get Predictions for test set AND format for Kaggle submission
predict_and_format(linear_workflow, test, "./linear_predictions.csv")
# 1.01231


# poisson regression ------------------------------------------------------

pois_mod <- poisson_reg() %>% 
  set_engine("glm") # glm = generalized linear model

pois_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(pois_mod) %>% 
  fit(data = log_train)

# Look at the fitted poisson model
extract_fit_engine(pois_workflow) %>%
  summary()

predict_and_format(pois_workflow, test, "./poisson_predictions.csv")
# 1.05155


# penalized regression ----------------------------------------------------

penalized_model <- linear_reg(penalty=0.0000000001, mixture=1) %>% # Set model and tuning
  set_engine("glmnet") 

# set workflow
penalized_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(penalized_model) %>%
  fit(data = log_train)

predict_and_format(penalized_wf, test, "./penalized_predictions.csv")
# 1.01029


# tuning ------------------------------------------------------------------

# Penalized regression model
penalized_model <- linear_reg(penalty=tune(),
                              mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

# Set Workflow
tuning_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(penalized_model)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

# split data for cross validation
folds <- vfold_cv(log_train, v = 5, repeats = 5)

## Run the CV
CV_results <- tuning_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae)) #Or leave metrics NULL

# plot results
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the Workflow & fit it
tuning_wf <- tuning_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=log_train)

predict_and_format(tuning_wf, test, "./penalized_tuned_predictions.csv")
# 1.00991


# regression trees --------------------------------------------------------

my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")

my_tree_recipe <- recipe(count ~ ., data = log_train) %>% 
  step_time(datetime, features=c("hour")) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% 
  step_num2factor(season, levels=c("spring", "summer", "fall", "winter")) %>%
  step_num2factor(weather, levels=c("partly_cloudy", "misty", "rainy")) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("no", "yes"))) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_numeric_predictors())

# Create a workflow with model & recipe
tree_wf <- workflow() %>%
  add_recipe(my_tree_recipe) %>%
  add_model(my_mod)

## Set up grid of tuning values
tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 6) ## L^2 total tuning possibilities

# Set up K-fold CV
folds <- vfold_cv(log_train, v = 5, repeats = 5)

tree_CV_results <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae)) #Or leave metrics NULL

# Find best tuning parameters
bestTune <- tree_CV_results %>%
  select_best("rmse")

# Finalize workflow and predict
final_wf <- tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=log_train)

predict_and_format(final_wf, test, "./reg_tree_predictions.csv")
# 0.48765

# random forests ----------------------------------------------------------


rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 1000) %>% # 500 or 1000
  set_engine("ranger") %>% 
  set_mode("regression")


# create workflow with model and recipe
## I'll try using my regression tree recipe first
rf_wf <- workflow() %>%
  add_recipe(my_tree_recipe) %>%
  add_model(rf_mod)

# grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1, (ncol(log_train)-1))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

# set up K-fold CV
folds <- vfold_cv(log_train, v = 5, repeats = 5)

rf_CV_results <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae)) #Or leave metrics NULL

# Find best tuning parameters
bestTune <- rf_CV_results %>%
  select_best("rmse")

# Finalize workflow and predict
final_wf <- rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=log_train)

predict_and_format(final_wf, test, "./random_forest_predictions.csv")
# 0.45272


# model stacking ----------------------------------------------------------

## Split data for CV
stacking_folds <- vfold_cv(train, v = 10, repeats=1)

## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5) ## L^2 total tuning possibilities

reg_tree_tuning_grid <- grid_regular(tree_depth(),
                                     cost_complexity(),
                                     min_n(),
                                     levels = 6) ## L^2 total tuning possibilities

## Run the CV
preg_models <- preg_wf %>%
  tune_grid(resamples=stacking_folds,
            grid=preg_tuning_grid,
            metrics=metric_set(rmse),
            control = untunedModel) # including the control grid in the tuning ensures you can
# call on it later in the stacked model

## Create other resampling objects with different ML algorithms to include in a stacked model, for example
lin_reg <-
  linear_reg() %>%
  set_engine("lm")
lin_reg_wf <-
  workflow() %>%
  add_model(lin_reg) %>%
  add_recipe(my_recipe)
lin_reg_model <-
  fit_resamples(
    lin_reg_wf,
    resamples = stacking_folds,
    metrics = metric_set(rmse),
    control = tunedModel
  )

# add regression tree
reg_tree <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")

reg_tree_wf <-
  workflow() %>%
  add_model(reg_tree) %>%
  add_recipe(my_tree_recipe)

reg_tree_model <-
  tune_grid(
    reg_tree_wf,
    grid = reg_tree_tuning_grid,
    resamples = stacking_folds,
    metrics = metric_set(rmse),
    control = untunedModel)

# random_forest_model <-
#   tune_grid(
#     rf_wf,
#     grid = tuning_grid,
#     resamples = folds,
#     metrics = metric_set(rmse),
#     control = untunedModel) # including this in my stack made the score go up


## Specify with models to include
my_stack <- stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(lin_reg_model) %>% 
  add_candidates(reg_tree_model)

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

## If you want to build your own metalearner you'll have to do so manually
## using
stackData <- as_tibble(my_stack)

## Use the stacked data to get a prediction
stack_mod %>% predict(new_data=test)


# Get Predictions for test set AND format for Kaggle
test_preds <- stack_mod %>% predict(new_data=test) %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

# Write prediction file to CSV
vroom_write(x=test_preds, file="./stacking_predictions.csv", delim=",")
# 0.74447


# bart --------------------------------------------------------------------

# create bart model and workflow
bart_model <- bart(trees = tune(),
                   prior_terminal_node_coef = tune(),
                   prior_terminal_node_expo = tune()) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

bart_workflow <-
  workflow(my_recipe) %>%
  add_model(bart_model)

# create tuning grid
tuning_grid <- grid_regular(trees(),
                            prior_terminal_node_coef(),
                            prior_terminal_node_expo(),
                            levels = 5)

# cross validation
cross_validation <- vfold_cv(data = log_train, v = 5, repeats = 1)

cross_validation_results <- bart_workflow %>%
  tune_grid(resamples = cross_validation,
    grid = 3,
    metrics=metric_set(rmse))

# Get the best hyperparameters
best_params <- cross_validation_results %>%
  select_best('rmse')

# use the best model
final_workflow_bart <- bart_workflow %>%
  finalize_workflow(best_params) %>%
  fit(data=log_train)

# predict and format for Kaggle
predict_and_format(final_workflow_bart, test, "./bart_predictions.csv")
# 0.39277







