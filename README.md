# NYC Property Sales Analysis

This repository contains an advanced analysis of New York City property sales data using multiple machine learning models.

## Project Structure

```
NYC_Final_Project/
├── data/               # Data directory
│   ├── nyc-rolling-sales.csv
│   └── NYC_Codes.csv
├── results/            # Model results and visualizations
├── docs/              # Documentation and reports
├── NYCPropertySales_Advanced.R    # Main analysis script
├── NYC_Property_Sales_Report.Rmd  # R Markdown report
└── README.md
```

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/[your-username]/NYC_Property_Sales.git
   cd NYC_Property_Sales
   ```

2. Required R packages will be automatically installed by the script, but you can install them manually:
   ```R
   install.packages(c("tidyverse", "caret", "xgboost", "randomForest", "lightgbm"))
   ```

3. The script will automatically download required datasets:
   - NYC Property Sales data
   - NYC Building Codes

4. Run the analysis:
   ```R
   source("NYCPropertySales_Advanced.R")
   ```

## Data Sources

- NYC Property Sales data: Originally from [NYC Department of Finance](https://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page)
- Building Classification Codes: [NYC Department of Finance](https://www.nyc.gov/assets/finance/jump/hlpbldgcode.html)

## Models

The analysis includes three machine learning models:
1. XGBoost
2. Random Forest
3. LightGBM

## Results

Results and model comparisons can be found in:
- `results/model_predictions.csv`
- `results/model_performance.csv`
- Generated visualizations in the `results` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details. 