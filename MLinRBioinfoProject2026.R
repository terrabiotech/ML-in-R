###Predicting Overweight Phenotypes in Cats Using a Synthetic dataset for Machine Learning in R###
##Comparing top 5 popular cat breeds to the American Shorthair cat###
#By: Terra K
#R version 4.4.3


# --------------------------------------------------
# Initial setup
# --------------------------------------------------
dir.create("~/MLinRBioinfoProject2026") #creates directory
setwd("/home/terra/MLinRBioinfoProject2026") #sets current working directory
print(file.exists("cats_dataset.csv")) #checks if file is in directory
cats <- read.csv("cats_dataset.csv", check.names = FALSE)

#loads and installs libraries and packages
library(tidyverse)
library(tidymodels)
library(vip)
library(ranger)

#sets starting point for R’s random number generator, so any random values you generate afterward can be reproduced exactly the same way next time.
set.seed(123)

#---------------------------------------------------
#1. Load and clean data
# --------------------------------------------------
#previews the structure of a dataframe and shows the # of rows, columns, column type, etc in a readable format
glimpse(cats)
names(cats) <- trimws(names(cats))

#converts columns into factors, using cats as a dataframe
cats <- cats %>%
  mutate(
    Breed  = factor(.data$Breed),
    Color  = factor(.data$Color),
    Gender = factor(.data$Gender)
  ) %>%
  drop_na(`Age (Years)`, `Weight (kg)`, Breed, Color, Gender) %>% #drops rows that are missing values
  rename( #renames columns 
    Age    = `Age (Years)`,
    Weight = `Weight (kg)`
  )

#2a. Create overweight target (top 25% within breed)
#groups cats dataframe by breed
cats <- cats %>%
  group_by(Breed) %>%
  mutate(
    weight_q75 = quantile(Weight, 0.75, na.rm = TRUE), #computes 75 percentile and flags overweight values
    Overweight = if_else(Weight >= weight_q75, "Yes", "No") #compares breeds to 75 percentile and decides yes for overweight at or above overweight value
  ) %>%
  ungroup() %>% #removes the grouping by Breed so later operations apply to the whole data frame, not within breeds
  mutate(Overweight = factor(Overweight, levels = c("No", "Yes"))) %>% #converts character vector Overweight into a yes/no factor
  select(-weight_q75) #removes the intermediate 75th percentile column

table(cats$Overweight) #converts how many cats are 'yes' and 'no' 
prop.table(table(cats$Overweight)) * 100 #converts yes/no to a percentage

# --------------------------------------------------
#2b. TOP 5 BREEDS + AMERICAN SHORTHAIR
# --------------------------------------------------
#creates new dataframe where each row is a combo of a Breed and an Overweight value
breed_stats <- cats %>%
  count(Breed, Overweight) %>%
  group_by(Breed) %>% #groups counts by breed so that totals are computed per breed
  mutate(
    breed_total = sum(n), #gives the total number of cats in that breed (both overweight and not)
    overweight_rate = n / breed_total #gives the proportion of overweight cats in that breed
  ) %>%
  filter(Overweight == "Yes") %>% #keeps only overweight rows
  ungroup() %>% #ungroups breeds
  arrange(desc(overweight_rate)) %>% #sorts breeds from highest to lowest overweight rate.
  mutate(
    rank = row_number(), #assigns rank 1 to the breed with the highest overweight rate, rank 2 to the next, etc
    overweight_pct = round(overweight_rate * 100, 1) #converts proportion into a percentage (rounded to 1 decimal place)
  )

#gets American Shorthair row (wherever it ranks); makes small table containing only the American Shorthair row, wherever it ranks in the list
ash_row <- breed_stats %>% filter(Breed == "American Shorthair")

#gets top 5 breeds
top5_breeds <- breed_stats %>% slice_head(n = 5)

# Combine: Top 5 + American Shorthair
top5_plus_ash <- bind_rows(top5_breeds, ash_row) %>%
  mutate(
    vs_ash_pct = round(overweight_pct - ash_row$overweight_pct[1], 1)
  ) %>%
  select(Breed, rank, overweight_pct, breed_total, n, vs_ash_pct) %>%
  distinct()

#prints labeled header and the combined table showing, for each of the top‑5 and American Shorthair breeds: Breed, rank by overweight rate, overweight_pct (percentage of overweight cats), breed_total (total cats in that breed), n (number overweight), vs_ash_pct (how much higher/lower than American Shorthair’s overweight percentage)
print("=== TOP 5 BREEDS + AMERICAN SHORTHAIR ===")
print(top5_plus_ash)

# --------------------------------------------------
#3. Train-test split
# --------------------------------------------------
#sets starting point for R’s random number generator, so any random values you generate afterward can be reproduced exactly the same way next time.
set.seed(123)

#cats_split is not a dataframe, it is a split object
cats_split <- initial_split(cats, prop = 0.8, strata = Overweight) #creates single train‑test split of the data ; prop = 0.8 means 80% of the cats go to the training set, 20% to the testing set; strata = Overweight means the split is stratified by the Overweight variable, so the proportion of yes/no overweight cats is roughly the same in both training and test sets
cats_train <- training(cats_split) #extracts the training set (80% of the cats) as a data frame
cats_test  <- testing(cats_split) #extracts the testing / hold‑out set (20% of the cats) as a data frame

# --------------------------------------------------
#4. Preprocessing recipe
#predict Overweight from Age, Gender, Breed, Color
# --------------------------------------------------
#gives predicted Overweight using the variables Age, Gender, Breed, and Color
overweight_recipe <- recipe(
  Overweight ~ Age + Gender + Breed + Color,
  data = cats_train #says that these variables live in the cats_train data frame
) %>%
  step_dummy(all_nominal_predictors()) %>% #converts nominal predictors into dummy (0/1) variables, 1 column per level (except a reference level)
  step_normalize(all_numeric_predictors()) #means “apply this to all predictor columns that are factors or character vectors,” which here is Gender, Breed, and Color

# --------------------------------------------------
#5. Random forest model spec
# --------------------------------------------------
rf_spec <- rand_forest( #creates specification for a random forest model
  mtry  = tune(), #mtry = the # of predictors randomly sampled at each split in a tree; setting it to tune() means it'll search over different values of mtry later (e.g., via tune_grid())
  trees = 500, #sets number of trees to 500 (a common default)
  min_n = tune() #minimal node size (smallest number of observations required in a terminal node) ; marking it tune() as means it'll search over different node‑size values
) %>%
  set_mode("classification") %>% #sets type of model
  set_engine("ranger", importance = "impurity") #set_engine("ranger") tells parsnip to use the ranger package “under the hood” to fit the random‑forest model; importance = "impurity" tells ranger to compute predictor importance scores based on how much each variable reduces impurity (Gini or entropy) when used in splits across the trees

# --------------------------------------------------
#6. Workflow
# --------------------------------------------------
rf_wf <- workflow() %>% #creates empty workflow object, which is just a container that will hold both: a model specification (e.g., random forest) and a preprocessing recipe (e.g., dummy coding, normalization)
  add_model(rf_spec) %>% #attaches random‑forest model specification (rf_spec) to the workflow
  add_recipe(overweight_recipe) #attaches the preprocessing recipe (overweight_recipe) to the workflow

# --------------------------------------------------
#7. Cross-validation and tuning
# --------------------------------------------------
#sets starting point for R’s random number generator, so any random values you generate afterward can be reproduced exactly the same way next time.
set.seed(123)
cats_folds <- vfold_cv(cats_train, v = 5, strata = Overweight) #runs 5‑fold cross‑validation to tune the random‑forest hyperparameters (mtry and min_n) and then selects the best settings to create a final model
#vfold_cv() creates 5‑fold cross‑validation folds from cats_train
#v = 5 means the training data are split into 5 parts; each part is used once as a validation set while the other 4 train the model
#strata = Overweight keeps the proportion of “Yes/No” overweight cats approximately the same in each fold, which is important for an imbalanced‑like outcome
#cats_folds is now a list of 5 train/validation splits, not a plain data frame


#defines a grid of hyperparameter values
rf_grid <- grid_regular(
  mtry(range = c(3L, 20L)),
  min_n(range = c(2L, 10L)),
  levels = 5
)

#tunes the random‑forest model via cross‑validation
rf_tuned <- tune_grid(
  rf_wf,
  resamples = cats_folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc, accuracy)
)

#looks at performance across hyperparameters
collect_metrics(rf_tuned)

#selects best by ROC AUC
best_params <- select_best(rf_tuned, metric = "roc_auc")
best_params

#finalizes workflow
rf_final_wf <- finalize_workflow(rf_wf, best_params)

# --------------------------------------------------
#8. Fit final model on full training data
# --------------------------------------------------
rf_final_fit <- fit(rf_final_wf, data = cats_train)

# --------------------------------------------------
#9. Evaluate on test set
# --------------------------------------------------
#generates test‑set predictions
rf_test_preds <- predict(rf_final_fit, cats_test, type = "prob") %>%
  bind_cols(predict(rf_final_fit, cats_test, type = "class")) %>%
  bind_cols(cats_test %>% select(Overweight)) %>%
  rename(pred_class = .pred_class)

#computes performance metrics
rf_metrics <- rf_test_preds %>% #metrics() computes classification metrics (like accuracy, sensitivity, specificity) using: truth = Overweight (true label) and testimate = pred_class (predicted label)
  metrics(truth = Overweight, estimate = pred_class)

# computes the ROC‑AUC (area under the ROC curve), using: truth = Overweight and .pred_Yes (the probability of class "Yes") as the score
rf_roc <- rf_test_preds %>%
  roc_auc(truth = Overweight, .pred_Yes)

#creates a confusion matrix comparing true vs. predicted classes
rf_conf <- rf_test_preds %>%
  conf_mat(truth = Overweight, estimate = pred_class)

#prints these summary tables so you can inspect: overall accuracy / other metrics, ROC‑AUC, and the full confusion matrix
rf_metrics
rf_roc
rf_conf

print(
  autoplot(rf_conf, type = "heatmap") + #draws confusion matrix as a heatmap where cell color corresponds to count (or proportion)
    scale_fill_gradient( #sets the color gradient from dark purple‑pink ("#A23B72") for low counts to lightblue for high counts
      low = "#A23B72",
      high = "lightblue" 
    ) +
    labs(title = "Confusion Matrix Heatmap") + #adds a title
    theme_minimal() #uses a clean, minimal ggplot2 theme
)


# --------------------------------------------------
#10. Variable importance
# --------------------------------------------------
#extracts the fitted random forest model
rf_fitted_model <- extract_fit_parsnip(rf_final_fit)$fit
#plot top 10 variable‑importance scores
vip(rf_fitted_model, 
    num_features = 10, 
    geom = "col", 
    aesthetics = list(fill = "seagreen3")) +  # Purple bars
  #customizes the plot; adds name and theme
  labs(title = "Top 10 Variable Importance") +
  theme_minimal()

# --------------------------------------------------
#11. Age effect plot 
# --------------------------------------------------
#creates an age‑only prediction grid (age_grid)
age_grid <- tibble(
  Age = as.integer(seq(min(cats$Age), max(cats$Age), length.out = 50)), #creates 50 evenly spaced age values from the minimum to maximum observed age in cats
  Gender = factor("Male", levels = levels(cats$Gender)), #fixes gender to "Male" for all rows (the first level of cats$Gender)
  Breed  = factor(as.character(levels(cats$Breed)[1]), levels = levels(cats$Breed)), #fixes 1 representative breed (the first level in cats$Breed), same for all rows
  Color  = factor(as.character(levels(cats$Color)[1]), levels = levels(cats$Color)) #fixes 1 representative color, same for all rows
)

#predicts probabilies along age grid
#predict feeds age_grid into the final fitted random forest workflow and gives predicted class probabilities for each row (e.g., .pred_No, .pred_Yes) ; bind_cols(age_grid) adds the Age (and fixed Gender/Breed/Color) columns back so you can link each probability to a specific age 
age_effect <- predict(rf_final_fit, new_data = age_grid, type = "prob") %>%
  bind_cols(age_grid)

#plots age effect curve
print(
  ggplot(age_effect, aes(x = Age, y = .pred_Yes)) + #plots Age on x‑axis and predicted probability of “Overweight = Yes” on y‑axis
    geom_line(color = "darkred") + #draws smooth line connecting the predicted probabilities across ages
    labs( #labels axes and gives a clear title: this curve shows how the model’s estimated obesity risk changes with age, assuming Gender, Breed, and Color are fixed at the chosen reference value
      x = "Age (years)",
      y = "Predicted P(Overweight = Yes)",
      title = "Effect of age on overweight probability (holding other traits fixed)"
    ) +
    theme_minimal()
)

