###Predicting Overweight Phenotypes in Cats Using a Synthetic dataset for Machine Learning in R###
##Comparing top 5 popular cat breeds to the American Shorthair cat###
#By: Terra K
#R version 4.4.3
###FOR FEMALE CATS --> OG PROJECT IS MALE CATS


# --------------------------------------------------
# Initial setup
# --------------------------------------------------
dir.create("~/MLinRBioinfoProject2026FEM") 
setwd("/home/terra/MLinRBioinfoProject2026FEM") 
print(file.exists("cats_dataset.csv"))  
cats <- read.csv("cats_dataset.csv", check.names = FALSE)


install.packages("tidyverse")
library(tidyverse)
install.packages("tidymodels")
library(tidymodels)
install.packages("vip")
library(vip)
install.packages("ranger")
library(ranger)

set.seed(123)

# --------------------------------------------------
#1. Load and clean data
# --------------------------------------------------
glimpse(cats)
names(cats) <- trimws(names(cats))

cats <- cats %>%
  mutate(
    Breed  = factor(Breed),
    Color  = factor(Color),
    Gender = factor(Gender)
  ) %>%
  drop_na(`Age (Years)`, `Weight (kg)`, Breed, Color, Gender) %>%
  rename(
    Age    = `Age (Years)`,
    Weight = `Weight (kg)`
  )

# --------------------------------------------------
#2a. Create overweight target (top 25% within breed)
# --------------------------------------------------
cats <- cats %>%
  group_by(Breed) %>%
  mutate(
    weight_q75 = quantile(Weight, 0.75, na.rm = TRUE),  
    Overweight = if_else(Weight >= weight_q75, "Yes", "No") 
  ) %>%
  ungroup() %>% 
  mutate(Overweight = factor(Overweight, levels = c("No", "Yes"))) %>% 
  select(-weight_q75)

table(cats$Overweight)
prop.table(table(cats$Overweight)) * 100

# --------------------------------------------------
#2b. TOP 5 BREEDS + AMERICAN SHORTHAIR
# --------------------------------------------------
breed_stats <- cats %>%
  count(Breed, Overweight) %>%
  group_by(Breed) %>% 
  mutate(
    breed_total = sum(n), 
    overweight_rate = n / breed_total 
  ) %>%
  filter(Overweight == "Yes") %>% 
  ungroup() %>% 
  arrange(desc(overweight_rate)) %>% 
  mutate(
    rank = row_number(),
    overweight_pct = round(overweight_rate * 100, 1
                          ) 
  )

ash_row <- breed_stats %>% filter(Breed == "American Shorthair")
top5_breeds <- breed_stats %>% slice_head(n = 5)

top5_plus_ash <- bind_rows(top5_breeds, ash_row) %>%
  mutate(
    vs_ash_pct = round(overweight_pct - ash_row$overweight_pct[1], 1)
  ) %>%
  select(Breed, rank, overweight_pct, breed_total, n, vs_ash_pct) %>%
  distinct()

print("=== TOP 5 BREEDS + AMERICAN SHORTHAIR ===")
print(top5_plus_ash)

# --------------------------------------------------
#3. Train-test split
# --------------------------------------------------
set.seed(123)

cats_split <- initial_split(cats, prop = 0.8, strata = Overweight)
cats_train <- training(cats_split)
cats_test  <- testing(cats_split)

# --------------------------------------------------
#4. Preprocessing recipe
# --------------------------------------------------
overweight_recipe <- recipe(
  Overweight ~ Age + Gender + Breed + Color,
  data = cats_train
) %>%
  step_relevel(Gender, ref_level = "Female") %>%   
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# --------------------------------------------------
#5. Random forest model spec
# --------------------------------------------------
rf_spec <- rand_forest(
  mtry  = tune(),
  trees = 500,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")

# --------------------------------------------------
#6. Workflow
# --------------------------------------------------
rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(overweight_recipe)

# --------------------------------------------------
#7. Cross-validation and tuning
# --------------------------------------------------
set.seed(123)

cats_folds <- vfold_cv(cats_train, v = 5, strata = Overweight)

rf_grid <- grid_regular(
  mtry(range = c(3L, 20L)),
  min_n(range = c(2L, 10L)),
  levels = 5
)

rf_tuned <- tune_grid(
  rf_wf,
  resamples = cats_folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc, accuracy)
)

collect_metrics(rf_tuned)

best_params <- select_best(rf_tuned, metric = "roc_auc")
best_params

rf_final_wf <- finalize_workflow(rf_wf, best_params)

# --------------------------------------------------
#8. Fit final model on full training data
# --------------------------------------------------
rf_final_fit <- fit(rf_final_wf, data = cats_train)

# --------------------------------------------------
#9. Evaluate on test set
# --------------------------------------------------
rf_test_preds <- predict(rf_final_fit, cats_test, type = "prob") %>%
  bind_cols(predict(rf_final_fit, cats_test, type = "class")) %>%
  bind_cols(cats_test %>% select(Overweight)) %>%
  rename(pred_class = .pred_class)

rf_metrics <- rf_test_preds %>%
  metrics(truth = Overweight, estimate = pred_class)

rf_roc <- rf_test_preds %>%
  roc_auc(truth = Overweight, .pred_Yes)

rf_conf <- rf_test_preds %>%
  conf_mat(truth = Overweight, estimate = pred_class)

rf_metrics
rf_roc
rf_conf

print(
  autoplot(rf_conf, type = "heatmap") +
    scale_fill_gradient(
      low = "#A23B72",
      high = "lightblue"
    ) +
    labs(title = "Confusion Matrix Heatmap") +
    theme_minimal()
)

# --------------------------------------------------
#10. Variable importance (clean labels)
# --------------------------------------------------
#(Dummy columns turned back into original variables after cleaning ex: Gender_Female -> Gender)
rf_fitted_model <- extract_fit_parsnip(rf_final_fit)$fit

vip_tbl <- vi(rf_fitted_model) %>%
  mutate(
    Variable = case_when(
      str_detect(Variable, "^Gender_") ~ "Gender",
      str_detect(Variable, "^Breed_")  ~ "Breed",
      str_detect(Variable, "^Color_")  ~ "Color",
      TRUE ~ Variable
    )
  ) %>%
  group_by(Variable) %>%
  summarise(Importance = sum(Importance), .groups = "drop") %>%
  arrange(desc(Importance))

print(
  ggplot(slice_head(vip_tbl, n = 10),
         aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_col(fill = "seagreen3") +
    coord_flip() +
    labs(
      title = "Top 10 Variable Importance",
      x = "Predictor",
      y = "Importance"
    ) +
    theme_minimal()
)

# --------------------------------------------------
#11. Age effect plot for FEMALES only
# --------------------------------------------------
age_grid <- tibble(
  Age = seq(min(cats$Age), max(cats$Age), length.out = 50),
  Gender = factor("Female", levels = levels(cats$Gender)),
  Breed  = factor(levels(cats$Breed)[1], levels = levels(cats$Breed)),
  Color  = factor(levels(cats$Color)[1], levels = levels(cats$Color))
)

age_effect <- predict(rf_final_fit, new_data = age_grid, type = "prob") %>%
  bind_cols(age_grid)

print(
  ggplot(age_effect, aes(x = Age, y = .pred_Yes)) +
    geom_line(color = "darkred", linewidth = 1.2) +
    labs(
      x = "Age (years)",
      y = "Predicted P(Overweight = Yes)",
      title = "Effect of Age on Overweight Probability for Female Cats"
    ) +
    theme_minimal()
)
