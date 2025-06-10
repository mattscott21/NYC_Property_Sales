# NYC Property Sales Analysis
# Author: Matthew Scott
#Github: https://github.com/mattscott21/NYC_Property_Sales

# ============================================================================
# 1. Setup and Dependencies
# ============================================================================

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Define required packages
required_packages <- c(
  # Data manipulation and analysis
  "tidyverse", "dplyr", "readr",
  
  # Visualization
  "ggplot2", "ggcorrplot", "gridExtra", "viridis", "scales",
  
  # Machine Learning
  "caret", "xgboost", "randomForest", "lightgbm",
  
  # Parallel processing
  "doParallel",
  
  # Data imputation
  "mice",
  
  # Reporting
  "knitr", "rmarkdown", "kableExtra",
  
  # Utilities
  "lubridate", "png", "grid", "DT", "moments"
)

# Install and load packages
for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Create results directory if it doesn't exist
if(!dir.exists("results")) dir.create("results")

# Set parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# ============================================================================
# 2. Data Loading and Initial Processing
# ============================================================================

# Load the data from GitHub
# Note: Using raw GitHub content URLs
nyc_rolling_sales <- read_csv("https://raw.githubusercontent.com/mattscott21/NYC_Property_Sales/main/data/nyc-rolling-sales.csv")
nyc_codes <- read_csv("https://raw.githubusercontent.com/mattscott21/NYC_Property_Sales/main/data/NYC_Codes.csv")

# ============================================================================
# 3. Data Preprocessing and Cleaning
# ============================================================================

# Data preprocessing
nyc_rolling_sales <- nyc_rolling_sales %>%
  # Add borough names and building codes
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
  left_join(nyc_codes, by = "Building Code") %>%

  # Remove unnecessary columns and convert numeric fields
  select(-c(1, `EASE-MENT`, `APARTMENT NUMBER`)) %>%
  mutate(
    `SALE PRICE` = as.numeric(gsub("[^0-9.]", "", `SALE PRICE`)),
    `GROSS SQUARE FEET` = as.numeric(gsub("[^0-9.]", "", `GROSS SQUARE FEET`)),
    `LAND SQUARE FEET` = as.numeric(gsub("[^0-9.]", "", `LAND SQUARE FEET`))
  ) %>%
  
  # Filter data
  # Filtering sales, sq ft, and year built to remove potential data entry errors
  filter(
    `SALE PRICE` >= 100000,  # Minimum meaningful sale price
    `GROSS SQUARE FEET` > 0 & `GROSS SQUARE FEET` < 999999999, 
    `LAND SQUARE FEET` > 0 & `LAND SQUARE FEET` < 999999999,  
    `YEAR BUILT` > 0,  
    Description %in% c("ONE FAMILY DWELLINGS", "TWO FAMILY DWELLINGS")  # Focus on residential properties
  )

# ============================================================================
# 4. Feature Engineering
# ============================================================================

# Feature engineering
# Used log transformations to handle the right-skewed distribution of prices and square footage
nyc_rolling_sales <- nyc_rolling_sales %>%
  mutate(
    # Log transformations
    log_sale_price = log(`SALE PRICE`),
    log_gross_sqft = log(`GROSS SQUARE FEET`),
    
    # Date features
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
    
    # Property features
    building_age = sale_year - `YEAR BUILT`,
    price_per_sqft = `SALE PRICE` / `GROSS SQUARE FEET`
  ) %>%
  select(-`SALE DATE`)

# Remove outliers using z-score
# Used a z-score threshold of 1.2816 (80th percentile) to remove extreme values
nyc_rolling_sales <- nyc_rolling_sales %>%
  group_by(NEIGHBORHOOD) %>%
  filter(abs(scale(log_sale_price)) <= 1.2816) %>%
  ungroup()

# ============================================================================
# 5. Statistical Analysis
# ============================================================================

# Calculate neighborhood statistics
# Statistics help capture local market conditions
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

# Calculate borough statistics
# Statistics help capture borough-level market conditions
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
    borough_skew = median(`SALE PRICE`, na.rm = TRUE)/mean(`SALE PRICE`, na.rm = TRUE)
  ) %>%
  mutate(
    borough_skew_median_price_interaction = borough_skew * borough_median_price,
    borough_skew_median_price_per_sqft_interaction = borough_skew * borough_median_price_per_sqft
  )

# Join statistics to main dataset
nyc_rolling_sales <- nyc_rolling_sales %>%
  left_join(neighborhood_stats, by = "NEIGHBORHOOD") %>%
  left_join(borough_stats, by = "borough_name") %>%
  mutate(
    neighborhood_id = as.integer(factor(NEIGHBORHOOD))/1000,
    building_age_neighborhood_interaction = building_age * neighborhood_avg_building_age,
    season_borough_interaction = interaction(season, borough_name),
    description_building_code_interaction = interaction(Description, `Building Code`)
  )

# ============================================================================
# 6. Dimensionality Reduction (SVD)
# ============================================================================

# Model preparation
features <- c(
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

# Prepare categorical variables
# These variables are converted to factors for proper model handling
categorical_vars <- c(
  "Building Code", "season", "borough_name", "Description",
  "season_borough_interaction"
)

# Prepare model data
model_data <- nyc_rolling_sales %>%
  select(all_of(c(features, categorical_vars))) %>%
  mutate(across(all_of(categorical_vars), as.factor))

# Perform SVD on numeric features
numeric_features <- model_data %>%
  select_if(is.numeric) %>%
  scale()  # Scale the features

# Perform SVD
svd_result <- svd(numeric_features)

# Calculate variance explained by each component
variance_explained <- (svd_result$d^2) / sum(svd_result$d^2)
cumulative_variance <- cumsum(variance_explained)

# Create scree plot
scree_data <- data.frame(
  Component = 1:length(variance_explained),
  Variance = variance_explained,
  Cumulative = cumulative_variance
)

# Plot variance explained
scree_plot <- ggplot(scree_data, aes(x = Component)) +
  geom_line(aes(y = Variance, color = "Individual")) +
  geom_line(aes(y = Cumulative, color = "Cumulative")) +
  geom_point(aes(y = Variance, color = "Individual")) +
  geom_point(aes(y = Cumulative, color = "Cumulative")) +
  labs(title = "Scree Plot of Variance Explained by SVD Components",
       x = "Component Number",
       y = "Proportion of Variance Explained",
       color = "Type") +
  theme_minimal() +
  scale_color_manual(values = c("Individual" = "blue", "Cumulative" = "red"))

# Create component contribution plot
component_contributions <- data.frame(
  Feature = colnames(numeric_features),
  PC1 = svd_result$v[, 1],
  PC2 = svd_result$v[, 2]
) %>%
  pivot_longer(cols = c(PC1, PC2), 
               names_to = "Component", 
               values_to = "Contribution")

# Plot feature contributions to first two components
contribution_plot <- ggplot(component_contributions, 
                           aes(x = Feature, y = Contribution, fill = Component)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Feature Contributions to First Two Principal Components",
       x = "Features",
       y = "Contribution") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8))

# Create biplot of first two components
biplot_data <- data.frame(
  PC1 = numeric_features %*% svd_result$v[, 1],
  PC2 = numeric_features %*% svd_result$v[, 2],
  Price = nyc_rolling_sales$`SALE PRICE`
)

biplot <- ggplot(biplot_data, aes(x = PC1, y = PC2, color = log(Price))) +
  geom_point(alpha = 0.5) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Biplot of First Two Principal Components",
       x = "First Principal Component",
       y = "Second Principal Component",
       color = "Log(Price)") +
  theme_minimal()

# Create correlation heatmap of selected important features
selected_features <- c(
  "SALE PRICE", "GROSS SQUARE FEET", "LAND SQUARE FEET", 
  "YEAR BUILT", "price_per_sqft", "building_age",
  "borough_avg_price", "borough_median_price",
  "neighborhood_avg_price", "neighborhood_median_price"
)

correlation_matrix <- cor(nyc_rolling_sales %>% select(all_of(selected_features)))
correlation_plot <- ggcorrplot(correlation_matrix,
                              method = "circle",
                              type = "lower",
                              lab = TRUE,
                              lab_size = 3,
                              colors = c("#6BAED6", "white", "#EE6A50"),
                              title = "Correlation Matrix of Key Features",
                              ggtheme = theme_minimal()) +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8)
  )

# Save all plots
ggsave("results/svd_scree_plot.png", scree_plot, width = 10, height = 6)
ggsave("results/svd_contribution_plot.png", contribution_plot, width = 12, height = 8)
ggsave("results/svd_biplot.png", biplot, width = 10, height = 6)
ggsave("results/correlation_heatmap.png", correlation_plot, width = 12, height = 10, dpi = 300)

# Determine number of components to keep (explaining 95% of variance)
n_components <- which(cumulative_variance >= 0.95)[1]

# Create reduced feature set using SVD
reduced_features <- numeric_features %*% svd_result$v[, 1:n_components]

# Add reduced features to model data
reduced_feature_names <- paste0("svd_comp_", 1:n_components)
colnames(reduced_features) <- reduced_feature_names

# Create feature contribution matrix with original feature names
feature_contributions <- svd_result$v[, 1:n_components]
rownames(feature_contributions) <- colnames(numeric_features)
colnames(feature_contributions) <- reduced_feature_names

# Get top contributing features for each component
top_features_per_component <- list()
for(i in 1:n_components) {
  # Get absolute contributions and sort
  contributions <- abs(feature_contributions[, i])
  top_features <- names(sort(contributions, decreasing = TRUE))[1:5]
  top_features_per_component[[i]] <- data.frame(
    Component = i,
    Feature = top_features,
    Contribution = contributions[top_features]
  )
}

# Combine with original categorical features
model_data_reduced <- cbind(
  as.data.frame(reduced_features),
  model_data %>% select(all_of(categorical_vars))
)

# Save SVD results with enhanced feature tracking
svd_results <- list(
  variance_explained = variance_explained,
  cumulative_variance = cumulative_variance,
  n_components = n_components,
  feature_contributions = feature_contributions,
  top_features_per_component = top_features_per_component,
  component_correlations = cor(numeric_features, reduced_features),
  original_feature_names = colnames(numeric_features)
)
saveRDS(svd_results, "results/svd_results.rds")

# Create and save feature contribution plot
feature_contribution_plot <- ggplot(
  do.call(rbind, top_features_per_component),
  aes(x = Component, y = Contribution, fill = Feature)
) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Component, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Top 5 Contributing Features for Each SVD Component",
    x = "SVD Component",
    y = "Absolute Contribution"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("results/feature_contribution_plot.png", feature_contribution_plot, width = 15, height = 10)

# ============================================================================
# 7. Model Training and Evaluation
# ============================================================================

# Split data with reduced features
set.seed(42)
train_index <- createDataPartition(nyc_rolling_sales$log_sale_price, p = 0.8, list = FALSE)
train_data <- nyc_rolling_sales[train_index, ]
test_data <- nyc_rolling_sales[-train_index, ]
X_train <- model_data_reduced[train_index, ]
X_test <- model_data_reduced[-train_index, ]

# Create target variables
y_train <- train_data$log_sale_price
y_test <- test_data$log_sale_price

# Prepare data for tree-based models
X_train_encoded <- model.matrix(~ . - 1, data = X_train)
X_test_encoded <- model.matrix(~ . - 1, data = X_test)
dtrain <- xgb.DMatrix(data = X_train_encoded, label = y_train)
dtest <- xgb.DMatrix(data = X_test_encoded, label = y_test)
dtrain_lgb <- lgb.Dataset(data = X_train_encoded, label = y_train)
dtest_lgb <- lgb.Dataset(data = X_test_encoded, label = y_test)

# Define model parameters
model_params <- list(
  XGBoost = list(
    objective = "reg:squarederror",
    eta = 0.1,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    nrounds = 1000
  ),
  RandomForest = list(
    ntree = 500,
    mtry = sqrt(ncol(X_train))
  ),
  LightGBM = list(
    objective = "regression",
    metric = "rmse",
    learning_rate = 0.1,
    num_leaves = 31,
    max_depth = 15,
    feature_fraction = 0.6,
    bagging_fraction = 0.8,
    lambda_l1 = 0.1,
    nrounds = 1000
  )
)

# Cross-validation to find optimal parameters
cv_control <- trainControl(method = "cv", number = 5)
cv_results <- list()

# XGBoost CV
cv_results$XGBoost <- xgb.cv(
  params = model_params$XGBoost,
  data = dtrain,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 100,
  verbose = 0
)

# LightGBM CV
cv_results$LightGBM <- lgb.cv(
  params = model_params$LightGBM,
  data = dtrain_lgb,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 100,
  verbose = 0
)

# Random Forest CV
cv_results$RandomForest <- train(
  x = X_train,
  y = y_train,
  method = "rf",
  trControl = cv_control,
  tuneGrid = expand.grid(mtry = seq(2, min(20, ncol(X_train)), by = 2))
)

# Update model parameters with CV results
model_params$XGBoost$nrounds <- cv_results$XGBoost$best_iteration
model_params$LightGBM$nrounds <- cv_results$LightGBM$best_iter
model_params$RandomForest$mtry <- cv_results$RandomForest$bestTune$mtry

# Train final models
xgb_model <- xgb.train(
  params = model_params$XGBoost,
  data = dtrain,
  nrounds = model_params$XGBoost$nrounds,
  watchlist = list(train = dtrain, test = dtest)
)

rf_model <- randomForest(
  x = X_train, 
  y = y_train,
  ntree = model_params$RandomForest$ntree,
  mtry = model_params$RandomForest$mtry,
  importance = TRUE
)

lgb_model <- lgb.train(
  params = model_params$LightGBM,
  data = dtrain_lgb,
  nrounds = model_params$LightGBM$nrounds,
  valids = list(test = dtest_lgb)
)

# Train linear regression model
train_data_lm <- as.data.frame(X_train_encoded)
train_data_lm$log_sale_price <- y_train
lm_model <- lm(log_sale_price ~ ., data = train_data_lm)

# Make predictions
xgb_predictions <- predict(xgb_model, dtest)
rf_predictions <- predict(rf_model, X_test)
lgb_predictions <- predict(lgb_model, X_test_encoded)
test_data_lm <- as.data.frame(X_test_encoded)
lm_predictions <- predict(lm_model, newdata = test_data_lm)

# Transform predictions
xgb_predictions_exp <- exp(xgb_predictions)
rf_predictions_exp <- exp(rf_predictions)
lgb_predictions_exp <- exp(lgb_predictions)
lm_predictions_exp <- exp(lm_predictions)
actual_values_exp <- exp(y_test)

# Create results dataframe
test_results <- data.frame(
  actual = actual_values_exp,
  xgb_predicted = xgb_predictions_exp,
  rf_predicted = rf_predictions_exp,
  lgb_predicted = lgb_predictions_exp,
  lm_predicted = lm_predictions_exp,
  borough = X_test$borough_name,
  desc = X_test$Description
)

# Calculate residuals
test_results$xgb_residuals <- test_results$actual - test_results$xgb_predicted
test_results$rf_residuals <- test_results$actual - test_results$rf_predicted
test_results$lgb_residuals <- test_results$actual - test_results$lgb_predicted
test_results$lm_residuals <- test_results$actual - test_results$lm_predicted

# Calculate metrics
# Note: We use RMSE, MedAE, and R² for comprehensive model evaluation
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  medae <- median(abs(actual - predicted))
  r2 <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(list(RMSE = rmse, MedAE = medae, R2 = r2))
}

# Calculate overall metrics for all models
model_names <- c("XGBoost", "Random Forest", "LightGBM", "Linear Regression")
prediction_cols <- c("xgb_predicted", "rf_predicted", "lgb_predicted", "lm_predicted")

overall_metrics <- data.frame(
  Model = model_names,
  RMSE = sapply(prediction_cols, function(col) {
    metrics <- calculate_metrics(test_results$actual, test_results[[col]])
    sprintf("$%s", format(round(metrics$RMSE), big.mark=","))
  }),
  MedAE = sapply(prediction_cols, function(col) {
    metrics <- calculate_metrics(test_results$actual, test_results[[col]])
    sprintf("$%s", format(round(metrics$MedAE), big.mark=","))
  }),
  R2 = sapply(prediction_cols, function(col) {
    metrics <- calculate_metrics(test_results$actual, test_results[[col]])
    metrics$R2
  })
)

# Save overall metrics
saveRDS(overall_metrics, "results/overall_metrics.rds")

# Calculate borough-specific metrics
borough_metrics_list <- list()
for(borough in unique(test_results$borough)) {
  borough_data <- test_results[test_results$borough == borough,]
  
  borough_metrics <- data.frame(
    Model = model_names,
    RMSE = sapply(prediction_cols, function(col) {
      metrics <- calculate_metrics(borough_data$actual, borough_data[[col]])
      sprintf("$%s", format(round(metrics$RMSE), big.mark=","))
    }),
    MedAE = sapply(prediction_cols, function(col) {
      metrics <- calculate_metrics(borough_data$actual, borough_data[[col]])
      sprintf("$%s", format(round(metrics$MedAE), big.mark=","))
    }),
    R2 = sapply(prediction_cols, function(col) {
      metrics <- calculate_metrics(borough_data$actual, borough_data[[col]])
      metrics$R2
    }),
    MAPE = sapply(prediction_cols, function(col) {
      mean(abs((borough_data$actual - borough_data[[col]]) / borough_data$actual)) * 100
    })
  )
  
  borough_metrics_list[[borough]] <- borough_metrics
  saveRDS(borough_metrics, sprintf("results/%s_metrics.rds", tolower(gsub(" ", "_", borough))))
}

# Save borough metrics
saveRDS(borough_metrics_list, "results/borough_metrics.rds")

# ============================================================================
# 8. Price Range Analysis
# ============================================================================

# Create price ranges
test_results$price_range <- cut(test_results$actual,
  breaks = c(0, 250000, 500000, 750000, 1000000, 1500000, 2000000, Inf),
  labels = c("Under 250K", "250K-500K", "500K-750K", "750K-1M", "1M-1.5M", "1.5M-2M", "Over 2M")
)

# Calculate price range metrics
price_range_metrics <- test_results %>%
  group_by(price_range) %>%
  summarise(
    # XGBoost metrics
    xgb_rmse = sqrt(mean((actual - xgb_predicted)^2)),
    xgb_medae = median(abs(actual - xgb_predicted)),
    xgb_r2 = 1 - sum((actual - xgb_predicted)^2) / sum((actual - mean(actual))^2),
    xgb_mape = mean(abs((actual - xgb_predicted) / actual)) * 100,
    
    # Random Forest metrics
    rf_rmse = sqrt(mean((actual - rf_predicted)^2)),
    rf_medae = median(abs(actual - rf_predicted)),
    rf_r2 = 1 - sum((actual - rf_predicted)^2) / sum((actual - mean(actual))^2),
    rf_mape = mean(abs((actual - rf_predicted) / actual)) * 100,
    
    # LightGBM metrics
    lgb_rmse = sqrt(mean((actual - lgb_predicted)^2)),
    lgb_medae = median(abs(actual - lgb_predicted)),
    lgb_r2 = 1 - sum((actual - lgb_predicted)^2) / sum((actual - mean(actual))^2),
    lgb_mape = mean(abs((actual - lgb_predicted) / actual)) * 100,
    
    # Linear Regression metrics
    lm_rmse = sqrt(mean((actual - lm_predicted)^2)),
    lm_medae = median(abs(actual - lm_predicted)),
    lm_r2 = 1 - sum((actual - lm_predicted)^2) / sum((actual - mean(actual))^2),
    lm_mape = mean(abs((actual - lm_predicted) / actual)) * 100
  )

# Save price range metrics
saveRDS(price_range_metrics, "results/price_range_metrics.rds")

# ============================================================================
# 9. Best Model Analysis
# ============================================================================

# Find best model for each borough
best_models_by_borough <- do.call(rbind, lapply(names(borough_metrics_list), function(borough) {
  metrics <- borough_metrics_list[[borough]]
  best_model <- metrics[which.min(metrics$MAPE), ]
  data.frame(
    borough = borough,
    model = best_model$Model,
    mape = best_model$MAPE
  )
}))

# Find best model for each price range
best_models_by_price <- price_range_metrics %>%
  pivot_longer(cols = ends_with("_mape"), names_to = "model", values_to = "mape") %>%
  group_by(price_range) %>%
  slice_min(mape) %>%
  mutate(model = gsub("_mape", "", model))

# Format MAPE values and transpose the table
mape_table <- price_range_metrics %>%
  select(price_range, ends_with("_mape")) %>%
  rename_with(~gsub("_mape", "", .), ends_with("_mape")) %>%
  mutate(across(ends_with("_mape"), ~sprintf("%.1f%%", .))) %>%
  pivot_longer(cols = -price_range, names_to = "model", values_to = "mape") %>%
  pivot_wider(names_from = price_range, values_from = mape)

# Save best models
write.csv(best_models_by_borough, "results/best_models_by_borough.csv", row.names = FALSE)
write.csv(best_models_by_price, "results/best_models_by_property_price.csv", row.names = FALSE)

# Create detailed best models dataframe
best_models_detailed <- test_results %>%
  group_by(borough, price_range) %>%
  summarise(
    xgb_mape = mean(abs((actual - xgb_predicted) / actual)) * 100,
    rf_mape = mean(abs((actual - rf_predicted) / actual)) * 100,
    lgb_mape = mean(abs((actual - lgb_predicted) / actual)) * 100,
    lm_mape = mean(abs((actual - lm_predicted) / actual)) * 100
  ) %>%
  pivot_longer(cols = ends_with("_mape"), names_to = "model", values_to = "mape") %>%
  group_by(borough, price_range) %>%
  slice_min(mape) %>%
  mutate(model = gsub("_mape", "", model))

# Save detailed best models
write.csv(best_models_detailed, "results/best_models_detailed.csv", row.names = FALSE)

# Define consistent theme
my_theme <- theme_bw() +
  theme(
    plot.title = element_text(size = 11, face = "bold", hjust = 0),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    strip.text = element_text(size = 10),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA),
    legend.key = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  )


# ============================================================================
# 10. Visualizations
# ============================================================================
# Create and save price distribution plots
price_dist_grid <- grid.arrange(
  # Original sale price histogram
  ggplot(nyc_rolling_sales, aes(x = `SALE PRICE`)) +
    geom_histogram(bins = 25, fill = "steelblue", alpha = 0.7) +
    scale_x_continuous(labels = scales::dollar) +
    labs(title = "Distribution of Sale Price",
         x = "Sale Price",
         y = "Count") +
    my_theme,
  
  # Log sale price histogram
  ggplot(nyc_rolling_sales, aes(x = log(`SALE PRICE`))) +
    geom_histogram(bins = 25, fill = "steelblue", alpha = 0.7) +
    scale_x_continuous(labels = scales::dollar) +
    labs(title = "Distribution of Log Sale Price",
         x = "Log Sale Price",
         y = "Count") +
    my_theme,
  
  # Original gross square feet histogram
  ggplot(nyc_rolling_sales, aes(x = `GROSS SQUARE FEET`)) +
    geom_histogram(bins = 25, fill = "darkgreen", alpha = 0.7) +
    scale_x_continuous(labels = scales::comma) +
    labs(title = "Distribution of Gross Square Feet",
         x = "Gross Square Feet",
         y = "Count") +
    my_theme,
  
  # Log gross square feet histogram
  ggplot(nyc_rolling_sales, aes(x = log(`GROSS SQUARE FEET`))) +
    geom_histogram(bins = 25, fill = "darkgreen", alpha = 0.7) +
    scale_x_continuous(labels = scales::comma) +
    labs(title = "Distribution of Log Gross Square Feet",
         x = "Log Gross Square Feet",
         y = "Count") +
    my_theme,
  ncol = 2
)
ggsave("results/price_distribution.png", price_dist_grid, width = 12, height = 10)
saveRDS(price_dist_grid, "results/price_distribution.rds")

# Create and save building age distribution plot
price_dist_plot <- ggplot(nyc_rolling_sales, aes(x = 'SALE PRICE', color = borough_name, fill = borough_name)) +
  geom_density(alpha = 0.3) +
  labs(title = "Price Distribution by Borough",
       x = "Sale Price",
       y = "Density",
       color = "Borough",
       fill = "Borough") +
  my_theme
ggsave("results/price_distribution.png", price_dist_plot, width = 12, height = 8)
saveRDS(price_dist_plot, "results/price_distribution.rds")


# Create and save building age distribution plot
age_dist_plot <- ggplot(nyc_rolling_sales, aes(x = building_age, color = borough_name, fill = borough_name)) +
  geom_density(alpha = 0.3) +
  labs(title = "Building Age Distribution by Borough",
       x = "Building Age (years)",
       y = "Density",
       color = "Borough",
       fill = "Borough") +
  my_theme
ggsave("results/building_age.png", age_dist_plot, width = 12, height = 8)
saveRDS(age_dist_plot, "results/building_age.rds")

# Create and save price per square foot plot
price_sqft_plot <- ggplot(nyc_rolling_sales, aes(x = borough_name, y = price_per_sqft, fill = borough_name)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_log10(labels = scales::dollar) +
  labs(title = "Price per Square Foot by Borough",
       x = "Borough",
       y = "Price per Square Foot (log scale)") +
  my_theme +
  theme(legend.position = "none")
ggsave("results/price_sqft.png", price_sqft_plot, width = 10, height = 6)
saveRDS(price_sqft_plot, "results/price_sqft.rds")

# Create and save seasonal patterns plot
seasonal_plot <- ggplot(nyc_rolling_sales, aes(x = season, y = `SALE PRICE`, fill = season)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_log10(labels = scales::dollar) +
  facet_wrap(~borough_name) +
  labs(title = "Seasonal Price Patterns by Borough",
       x = "Season",
       y = "Sale Price (log scale)") +
  my_theme +
  theme(legend.position = "none")
ggsave("results/seasonal_patterns.png", seasonal_plot, width = 12, height = 8)
saveRDS(seasonal_plot, "results/seasonal_patterns.rds")

# Create and save performance metrics plot
performance_metrics_plot <- ggplot(overall_metrics, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  labs(title = "Model Performance Comparison",
       x = "Model",
       y = "RMSE") +
  my_theme +
  theme(legend.position = "none")
ggsave("results/performance_metrics.png", performance_metrics_plot, width = 10, height = 6)

# Create and save borough performance plot
borough_performance_plot <- ggplot(test_results, aes(x = borough, y = xgb_residuals, fill = borough)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Model Performance by Borough",
       x = "Borough",
       y = "Residuals") +
  my_theme +
  theme(legend.position = "none")
ggsave("results/borough_performance.png", borough_performance_plot, width = 10, height = 6)

# Create and save price range performance plot
price_range_performance_plot <- ggplot(test_results, aes(x = price_range, y = xgb_residuals, fill = price_range)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Model Performance by Price Range",
       x = "Price Range",
       y = "Residuals") +
  my_theme +
  theme(legend.position = "none")
ggsave("results/property_price_performance.png", price_range_performance_plot, width = 10, height = 6)

# Create and save residual plots for each model
# XGBoost residuals
xgb_residual_plot <- ggplot(test_results, aes(x = xgb_predicted, y = xgb_residuals)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  labs(title = "XGBoost Residuals",
       x = "Predicted Price",
       y = "Residuals") +
  my_theme
ggsave("results/xgb_residual_plot.png", xgb_residual_plot, width = 10, height = 6)

# LightGBM residuals
lgb_residual_plot <- ggplot(test_results, aes(x = lgb_predicted, y = lgb_residuals)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  labs(title = "LightGBM Residuals",
       x = "Predicted Price",
       y = "Residuals") +
  my_theme
ggsave("results/lgb_residual_plot.png", lgb_residual_plot, width = 10, height = 6)

# Random Forest residuals
rf_residual_plot <- ggplot(test_results, aes(x = rf_predicted, y = rf_residuals)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  labs(title = "Random Forest Residuals",
       x = "Predicted Price",
       y = "Residuals") +
  my_theme
ggsave("results/rf_residual_plot.png", rf_residual_plot, width = 10, height = 6)

# Linear Regression residuals
lm_residual_plot <- ggplot(test_results, aes(x = lm_predicted, y = lm_residuals)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  labs(title = "Linear Regression Residuals",
       x = "Predicted Price",
       y = "Residuals") +
  my_theme
ggsave("results/lm_residual_plot.png", lm_residual_plot, width = 10, height = 6)

# Create and save price density plot by borough
price_density_plot <- ggplot(nyc_rolling_sales, aes(x = `SALE PRICE`, fill = borough_name)) +
  geom_density(alpha = 0.5) +
  scale_x_continuous(labels = scales::dollar, trans = "log10") +
  labs(title = "Distribution of Sale Prices by Borough",
       x = "Sale Price (log scale)",
       y = "Density",
       fill = "Borough") +
  my_theme +
  theme(legend.position = "right")

# Save the plot
ggsave("results/price_density_by_borough.png", price_density_plot, width = 12, height = 8)
saveRDS(price_density_plot, "results/price_density_by_borough.rds")

# Extract feature importance from XGBoost model
xgb_importance <- xgb.importance(feature_names = colnames(X_train_encoded), model = xgb_model)
xgb_importance <- as.data.frame(xgb_importance)
colnames(xgb_importance) <- c("Feature", "Gain", "Cover", "Frequency")

# Save feature importance
write.csv(xgb_importance, "results/xgboost_feature_importance.csv", row.names = FALSE)

# Extract feature importance from Random Forest model
rf_importance <- as.data.frame(importance(rf_model))
rf_importance$Feature <- rownames(rf_importance)
rf_importance <- rf_importance %>%
  select(Feature, `%IncMSE`, IncNodePurity) %>%
  rename(Gain = `%IncMSE`, Cover = IncNodePurity) %>%
  mutate(Frequency = 0)  # Random Forest doesn't provide frequency

# Save Random Forest feature importance
write.csv(rf_importance, "results/rf_feature_importance.csv", row.names = FALSE)

# Extract feature importance from LightGBM model
lgb_importance <- lgb.importance(lgb_model)
lgb_importance <- as.data.frame(lgb_importance)
colnames(lgb_importance) <- c("Feature", "Gain", "Cover", "Frequency")

# Save LightGBM feature importance
write.csv(lgb_importance, "results/lgb_feature_importance.csv", row.names = FALSE)

# Create and save price density plot by borough
price_density_plot <- ggplot(nyc_rolling_sales, aes(x = `SALE PRICE`, fill = borough_name)) +
  geom_density(alpha = 0.5) +
  scale_x_continuous(labels = scales::dollar, trans = "log10") +
  labs(title = "Distribution of Sale Prices by Borough",
       x = "Sale Price (log scale)",
       y = "Density",
       fill = "Borough") +
  my_theme +
  theme(legend.position = "right")

# Save the plot
ggsave("results/price_density_by_borough.png", price_density_plot, width = 12, height = 8)
saveRDS(price_density_plot, "results/price_density_by_borough.rds")