# Chinese Automotive Sales Forecasting Project

This project builds an end-to-end forecasting pipeline using SQL and Python to predict monthly vehicle sales in the Chinese automotive market. The workflow includes SQL-based data cleaning, feature engineering, model training, evaluation, visualisation, and analytical SQL summaries.

---

## Project Structure

Retail_Forcasting/
│
├── Data/
│   ├── raw_sales.csv
│   ├── translations.csv
│   ├── clean_sales_final.csv
│   ├── final_predictions_detailed.csv
│   ├── model_scoring_summary.csv
│   └── plots/
│
├── Notebooks/
│   ├── 01_Load_And_Explore_Raw_Data.ipynb
│   ├── 02_SQL_Setup.ipynb
│   ├── 03_SQL_Cleaning_ETL.ipynb
│   ├── 04_Load_Clean_And_Features.ipynb
│   ├── 05_Group_Preparation.ipynb
│   ├── 06_Train_Models.ipynb
│   ├── 07_Visualise_Results.ipynb
│   └── 08_SQL_Insights_And_Summary.ipynb
│
└── Forcasting_Main_Pipeline.py

---

## What the Project Does

1. **SQL ETL Cleaning**
   - Translates manufacturer names using a lookup table.
   - Normalises inconsistent date formats.
   - Removes invalid or missing fields.
   - Converts sales figures to numeric.
   - Deduplicates entries using SQL window functions.
   - Produces a clean table: `clean_sales_final`.

2. **Feature Engineering**
   - Generates time-based features: Year, Month, Month_Since_Start.
   - Groups data by (Make, Body_Type).
   - Filters out groups with fewer than 16 months of data.

3. **Model Training**
   - Trains a separate Gradient Boosting model for each group.
   - Uses one-hot encoding for categorical features.
   - Performs an 80/10/10 Train/Validation/Test split.
   - Saves results to `final_predictions_detailed.csv`.

4. **Evaluation**
   - Calculates MAE and RMSE for each group.
   - Saves scoring to `model_scoring_summary.csv`.

5. **Visualisation**
   - Plots actual vs predicted sales.
   - Highlights Train, Validation, and Test periods.
   - Saves images in `Data/plots/`.

6. **SQL Insights**
   - Identifies top manufacturers by total sales.
   - Shows industry-wide monthly trends.
   - Calculates month-on-month growth using SQL window functions.

---

## Skills Demonstrated
- SQL data cleaning and transformation
- Python data analysis and modelling
- Feature engineering for time-series forecasting
- Gradient Boosting regression
- Data visualisation
- Project structuring and reproducibility
- Working with messy real-world datasets

---

## Main Pipeline Script

The full workflow is automated in:
`Forcasting_Main_Pipeline.py`

This script:
- Loads raw data
- Runs SQL ETL cleaning
- Performs feature engineering
- Trains models for each vehicle group
- Generates predictions and plots
- Saves evaluation results
- Executes final SQL insights

---

This project forms part of my data analyst portfolio and demonstrates my ability to handle data end-to-end, from raw ingestion to forecasting and analysis.
