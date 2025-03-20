# NYC Property Sales Advanced Analysis
# This script performs advanced machine learning analysis on NYC property sales data
# Author: [Your Name]
# Date: 2024

#----Package Management----
# List of required packages
packages <- c(
  "tidyverse",
  "caret",
  "readr",
  "ggplot2",
  "dplyr",
  "ggcorrplot",
  "xgboost",
  "lubridate",
  "gridExtra",
  "randomForest",
  "doParallel",
  "lightgbm",
  "scales",
  "rvest",
  "data.table"
)

# Install and load required packages
for (package in packages) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package)
    library(package, character.only = TRUE)
  }
}

#----Data Download and Import----
# Create directories if they don't exist
for (dir in c("data", "results")) {
  if (!dir.exists(dir)) {
    dir.create(dir)
  }
}

# Function to download and extract data from GitHub
download_from_github <- function() {
  # GitHub raw URLs for the datasets
  github_base <- "https://raw.githubusercontent.com/your-username/NYC_Property_Sales/main/data"
  sales_url <- paste0(github_base, "/nyc-rolling-sales.csv")
  codes_url <- paste0(github_base, "/NYC_Codes.csv")
  
  tryCatch({
    # Download NYC sales data
    message("Downloading NYC Property Sales dataset...")
    download.file(sales_url, "data/nyc-rolling-sales.csv", mode = "wb")
    
    # Download building codes
    message("Downloading NYC building codes...")
    download.file(codes_url, "data/NYC_Codes.csv", mode = "wb")
    
    return(TRUE)
  }, error = function(e) {
    # If GitHub download fails, try scraping building codes and provide instructions for sales data
    message("Error downloading from GitHub: ", e$message)
    
    # Try to create building codes file from NYC website
    if (!file.exists("data/NYC_Codes.csv")) {
      message("\nAttempting to create building codes file from NYC website...")
      tryCatch({
        url <- "https://www.nyc.gov/assets/finance/jump/hlpbldgcode.html"
        webpage <- read_html(url)
        
        codes <- webpage %>%
          html_nodes("td:nth-child(1)") %>%
          html_text()
        
        descriptions <- webpage %>%
          html_nodes("td:nth-child(2)") %>%
          html_text()
        
        nyc_codes <- data.frame(
          `Building Code` = substr(codes, 1, 1),
          Description = trimws(descriptions)
        ) %>%
          distinct()
        
        write.csv(nyc_codes, "data/NYC_Codes.csv", row.names = FALSE)
        message("Successfully created building codes file")
      }, error = function(e) {
        message("Error creating building codes file: ", e$message)
      })
    }
    
    # Provide instructions for manual data download
    if (!file.exists("data/nyc-rolling-sales.csv")) {
      message("\nPlease obtain the NYC Property Sales dataset through one of these methods:")
      message("1. Download from the course GitHub repository:")
      message("   https://github.com/your-username/NYC_Property_Sales")
      message("2. Or download from Kaggle:")
      message("   https://www.kaggle.com/datasets/new-york-city/nyc-property-sales")
      message("\nPlace the file 'nyc-rolling-sales.csv' in the 'data' directory")
      return(FALSE)
    }
  })
}

# Check if data files exist, if not download them
if (!all(file.exists("data/nyc-rolling-sales.csv", "data/NYC_Codes.csv"))) {
  if (!download_from_github()) {
    stop("Required dataset(s) missing. Please follow the instructions above.")
  }
}

# Load datasets
nyc_rolling_sales <- read_csv("data/nyc-rolling-sales.csv")
nyc_codes <- read_csv("data/NYC_Codes.csv")

#----Data Preparation----
# Add Borough Names and Building Codes
nyc_rolling_sales <- nyc_rolling_sales %>%
  mutate(
    borough_name = case_when(
      BOROUGH == 1 ~ "MANHATTAN",
      BOROUGH == 2 ~ "BRONX",
      BOROUGH == 3 ~ "BROOKLYN",
      BOROUGH == 4 ~ "QUEENS",
      BOROUGH == 5 ~ "STATEN ISLAND"
    ),
    `Building Code` = substring(`BUILDING CLASS AT PRESENT`, 1, 1)
  ) %>%
  left_join(nyc_codes, by = "Building Code")

# Data Preprocessing
nyc_rolling_sales <- nyc_rolling_sales %>%
  select(-c(1, `EASE-MENT`, `APARTMENT NUMBER`)) %>%
  mutate(
    `SALE PRICE` = as.numeric(gsub("[^0-9.]", "", `SALE PRICE`)),
    `GROSS SQUARE FEET` = as.numeric(gsub("[^0-9.]", "", `GROSS SQUARE FEET`)),
    `LAND SQUARE FEET` = as.numeric(gsub("[^0-9.]", "", `LAND SQUARE FEET`))
  ) %>%
  filter(
    `SALE PRICE` >= 100000,
    `GROSS SQUARE FEET` > 0 & `GROSS SQUARE FEET` < 999999999,
    `LAND SQUARE FEET` > 0 & `LAND SQUARE FEET` < 999999999,
    `YEAR BUILT` > 0,
    `BOROUGH` > 1,
    Description %in% c("ONE FAMILY DWELLINGS", "TWO FAMILY DWELLINGS")
  )

#----Feature Engineering----
nyc_rolling_sales <- nyc_rolling_sales %>%
  mutate(
    log_sale_price = log(`SALE PRICE`),
    log_gross_sqft = log(`GROSS SQUARE FEET`),
    sale_date = as.Date(`SALE DATE`),
    sale_month = month(sale_date),
    sale_year = year(sale_date),
    is_weekend = if_else(wday(sale_date) %in% c(1, 7), 1, 0),
    season = factor(case_when(
      sale_month %in% c(12, 1, 2) ~ "Winter",
      sale_month %in% c(3, 4, 5) ~ "Spring",
      sale_month %in% c(6, 7, 8) ~ "Summer",
      sale_month %in% c(9, 10, 11) ~ "Fall"
    ), levels = c("Spring", "Summer", "Fall", "Winter")),
    building_age = sale_year - `YEAR BUILT`,
    price_per_sqft = `SALE PRICE` / `GROSS SQUARE FEET`
  ) %>%
  select(-`SALE DATE`)

# Filter outliers based on log sale price
nyc_rolling_sales <- nyc_rolling_sales %>%
  group_by(borough_name) %>%
  filter(abs(scale(log_sale_price)) <= 1.25) %>%
  ungroup()

#----Neighborhood and Borough Statistics----
# Calculate neighborhood-level statistics
neighborhood_stats <- nyc_rolling_sales %>%
  group_by(NEIGHBORHOOD) %>%
  summarise(
    neighborhood_avg_price = mean(`SALE PRICE`, na.rm = TRUE),
    neighborhood_median_price = median(`SALE PRICE`, na.rm = TRUE),
    neighborhood_avg_price_per_sqft = mean(price_per_sqft, na.rm = TRUE),
    neighborhood_median_price_per_sqft = median(price_per_sqft, na.rm = TRUE),
    neighborhood_avg_sqft = mean(`GROSS SQUARE FEET`, na.rm = TRUE),
    neighborhood_median_sqft = median(`GROSS SQUARE FEET`, na.rm = TRUE),
    neighborhood_avg_building_age = mean(building_age, na.rm = TRUE),
    neighborhood_median_building_age = median(building_age, na.rm = TRUE),
    neighborhood_sales = n(),
    neighborhood_skew = median(`SALE PRICE`, na.rm = TRUE)/mean(`SALE PRICE`, na.rm = TRUE)
  )

# Calculate borough-level statistics
borough_stats <- nyc_rolling_sales %>%
  group_by(borough_name) %>%
  summarise(
    borough_avg_price = mean(`SALE PRICE`, na.rm = TRUE),
    borough_median_price = median(`SALE PRICE`, na.rm = TRUE),
    borough_avg_price_per_sqft = mean(price_per_sqft, na.rm = TRUE),
    borough_median_price_per_sqft = median(price_per_sqft, na.rm = TRUE),
    borough_avg_sqft = mean(`GROSS SQUARE FEET`, na.rm = TRUE),
    borough_median_sqft = median(`GROSS SQUARE FEET`, na.rm = TRUE),
    borough_avg_building_age = mean(building_age, na.rm = TRUE),
    borough_median_building_age = median(building_age, na.rm = TRUE),
    borough_sales = n(),
    borough_skew = median(`SALE PRICE`, na.rm = TRUE)/mean(`SALE PRICE`, na.rm = TRUE),
    borough_skew_median_price_interaction = borough_skew * borough_median_price,
    borough_skew_median_price_per_sqft_interaction = borough_skew * borough_median_price_per_sqft
  )

# Join statistics back to main dataset
nyc_rolling_sales <- nyc_rolling_sales %>%
  left_join(neighborhood_stats, by = "NEIGHBORHOOD") %>%
  left_join(borough_stats, by = "borough_name")

# Create neighborhood ID and additional features
nyc_rolling_sales <- nyc_rolling_sales %>%
  mutate(
    neighborhood_id = as.integer(factor(NEIGHBORHOOD))/1000,
    building_age_neighborhood_interaction = building_age * neighborhood_avg_building_age,
    season_borough_interaction = interaction(season, borough_name),
    description_building_code_interaction = interaction(Description, `Building Code`)
  )

#----Model Preparation----
# Define features for modeling
features_to_use <- c(
  "BOROUGH", "RESIDENTIAL UNITS", "COMMERCIAL UNITS", "TOTAL UNITS", 
  "sale_month", "sale_year", "building_age", "ZIP CODE", "log_gross_sqft", 
  "BLOCK", "LOT", "neighborhood_avg_price", "neighborhood_median_price",
  "neighborhood_avg_price_per_sqft", "neighborhood_avg_building_age", 
  "neighborhood_avg_sqft", "neighborhood_median_sqft",
  "neighborhood_median_price_per_sqft", "neighborhood_median_building_age",
  "neighborhood_sales", "neighborhood_id", "neighborhood_skew",
  "borough_avg_price", "borough_median_price", "borough_avg_price_per_sqft",
  "borough_avg_building_age", "borough_avg_sqft", "borough_median_sqft",
  "borough_median_price_per_sqft", "borough_median_building_age",
  "borough_sales", "borough_skew", "building_age_neighborhood_interaction",
  "borough_skew_median_price_interaction",
  "borough_skew_median_price_per_sqft_interaction"
)

categorical_vars <- c(
  "Building Code", "season", "borough_name", "Description",
  "season_borough_interaction"
)

# Prepare model data
model_data <- nyc_rolling_sales %>%
  select(all_of(c(features_to_use, categorical_vars))) %>%
  mutate(across(all_of(categorical_vars), as.factor))

# Split data into training and testing sets
set.seed(42)
train_index <- createDataPartition(nyc_rolling_sales$log_sale_price, p = 0.8, list = FALSE)
train_data <- nyc_rolling_sales[train_index, ]
test_data <- nyc_rolling_sales[-train_index, ]
X_train <- model_data[train_index, ]
X_test <- model_data[-train_index, ]
y_train <- nyc_rolling_sales$log_sale_price[train_index]
y_test <- nyc_rolling_sales$log_sale_price[-train_index]

#----XGBoost Model----
# Prepare data for XGBoost
X_train_encoded <- model.matrix(~ . - 1, data = X_train)
X_test_encoded <- model.matrix(~ . - 1, data = X_test)
dtrain <- xgb.DMatrix(data = X_train_encoded, label = y_train)
dtest <- xgb.DMatrix(data = X_test_encoded, label = y_test)

# Set XGBoost parameters
params <- list(
  objective = "reg:pseudohubererror",
  booster = "gbtree",
  eta = 0.01,
  max_depth = 15,
  subsample = 0.8,
  colsample_bytree = 0.9,
  colsample_bylevel = 0.9,
  colsample_bynode = 0.9,
  lambda = 1,
  alpha = .1,
  gamma = 0,
  min_clild_weight = .8,
  eval_metric = "rmse",
  tree_method = "hist",
  maximize = FALSE
)

# Train XGBoost model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 10000,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10
)

#----Random Forest Model----
# Train Random Forest model
rf_model <- randomForest(
  x = X_train, 
  y = y_train,
  ntree = 1000,
  mtry = floor(sqrt(ncol(X_train))),
  importance = TRUE,
  maxnodes = 200,
  nodesize = 5,
  sampsize = floor(0.8 * nrow(X_train))
)

#----LightGBM Model----
# Prepare data for LightGBM
dtrain_lgb <- lgb.Dataset(data = X_train_encoded, label = y_train)
dtest_lgb <- lgb.Dataset(data = X_test_encoded, label = y_test)

# Set LightGBM parameters
lgb_params <- list(
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.1,
  num_leaves = 31,
  max_depth = 15,
  feature_fraction = 0.6,
  bagging_fraction = 0.8,
  lambda_l1 = 0.1,
  lambda_l2 = 0,
  verbosity = 1
)

# Train LightGBM model
lgb_model <- lgb.train(
  params = lgb_params,
  data = dtrain_lgb,
  nrounds = 10000,
  valids = list(test = dtest_lgb),
  early_stopping_rounds = 50
)

#----Model Predictions and Evaluation----
# Make predictions
xgb_predictions <- predict(xgb_model, dtest)
rf_predictions <- predict(rf_model, X_test)
lgb_predictions <- predict(lgb_model, X_test_encoded)

# Transform predictions back to original scale
xgb_predictions_exp <- exp(xgb_predictions)
rf_predictions_exp <- exp(rf_predictions)
lgb_predictions_exp <- exp(lgb_predictions)
actual_values_exp <- exp(y_test)

# Create results dataframe
test_results <- data.frame(
  actual = actual_values_exp,
  xgb_predicted = xgb_predictions_exp,
  rf_predicted = rf_predictions_exp,
  lgb_predicted = lgb_predictions_exp,
  borough = X_test$borough_name,
  desc = X_test$Description
)

# Calculate performance metrics
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- median(abs(actual - predicted))
  r2 <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(list(RMSE = rmse, MedAE = mae, R2 = r2))
}

# Calculate metrics for each model
xgb_metrics <- calculate_metrics(actual_values_exp, xgb_predictions_exp)
rf_metrics <- calculate_metrics(actual_values_exp, rf_predictions_exp)
lgb_metrics <- calculate_metrics(actual_values_exp, lgb_predictions_exp)

# Print results
cat("\nXGBoost Metrics:\n")
print(xgb_metrics)
cat("\nRandom Forest Metrics:\n")
print(rf_metrics)
cat("\nLightGBM Metrics:\n")
print(lgb_metrics)

# Save results
results_dir <- "results"
dir.create(results_dir, showWarnings = FALSE)
write.csv(test_results, file.path(results_dir, "model_predictions.csv"), row.names = FALSE)

# Create performance summary
model_performance <- data.frame(
  Model = c("XGBoost", "Random Forest", "LightGBM"),
  RMSE = c(xgb_metrics$RMSE, rf_metrics$RMSE, lgb_metrics$RMSE),
  MedAE = c(xgb_metrics$MedAE, rf_metrics$MedAE, lgb_metrics$MedAE),
  R2 = c(xgb_metrics$R2, rf_metrics$R2, lgb_metrics$R2)
)

write.csv(model_performance, file.path(results_dir, "model_performance.csv"), row.names = FALSE) 