##
## Bike Share EDA Code
##
 
# Libraries
library(tidyverse)
library(vroom)
library(patchwork)

# Read in the Data
bikes <- vroom("./train.csv")

# DataExplorer::plot_intro(bikes)
# head(bikes)

# convert variables to factors
bikes$season <- as.factor(bikes$season)
bikes$holiday <- as.factor(bikes$holiday)
bikes$workingday <- as.factor(bikes$workingday)
bikes$weather <- as.factor(bikes$weather)


# scatter plot of datetime and temp ---------------------------------------

plot_1 <- bikes %>% 
  ggplot(mapping = aes(x = temp, y = count)) +
  geom_point() +
  geom_smooth(se = F) +
  labs(title = "Temperature vs Count")

# scatter plot of datetime and rentals (with temp) ------------------------

plot_2 <- bikes %>% 
  ggplot(mapping = aes(x = datetime, y = count, color = temp)) +
  geom_point() +
  labs(title = "Rentals over time",
       subtitle = "With Temperature")

# Box plot of seasons and count -------------------------------------------

plot_3 <- bikes %>% 
  ggplot(mapping = aes(x = season, y = count)) +
  geom_boxplot() +
  labs(title = "Rentals Across the Seasons")


# Violin Plot of Count and Workingday across all seasons ------------------

plot_4 <- bikes %>% 
  ggplot(mapping = aes(x = workingday, y = count, color = season)) +
  geom_violin() +
  coord_flip() +
  labs(title = "Number of Rentals on Workingdays vs Weekends")

# Combined Plot -----------------------------------------------------------
(plot_1 + plot_2) / (plot_3 + plot_4)








