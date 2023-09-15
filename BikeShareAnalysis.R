##
## Bike Share Analysis Code
##

# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)

# Read in the Data
bikes <- vroom("./train.csv")

# cleaning ----------------------------------------------------------------

# convert variables to factors
bikes$season <- as.factor(bikes$season)
bikes$holiday <- as.factor(bikes$holiday)
bikes$workingday <- as.factor(bikes$workingday)
bikes$weather <- as.factor(bikes$weather)

# turn 4s into 3s
bikes$weather <- ifelse(bikes$weather == 4, 3, bikes$weather)


# feature engineering -----------------------------------------------------

my_recipe <- recipe(count ~ ., bikes) %>% 
  step_date(datetime, features = "dow") %>% # gets day of week from datetime
  step_rm(casual, registered, atemp) %>%  # remove casual and registered columns
  step_select(datetime, datetime_dow, season, holiday, workingday,
              weather, temp, humidity, windspeed, count) # reorder columns

prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, bikes)
