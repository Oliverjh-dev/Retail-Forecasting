import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import warnings

warnings.filterwarnings("ignore")

# -----------------------
# Paths (LOCAL ABSOLUTE PATHS FOR VS CODE)
# -----------------------

DATA_DIR = r"C:\Users\olive\PythonProjects\Retail_Forcasting\Data"

RAW_CSV_PATH = os.path.join(DATA_DIR, "raw_sales.csv")
TRANSLATION_CSV_PATH = os.path.join(DATA_DIR, "translations.csv")
SQL_DB_PATH = os.path.join(DATA_DIR, "auto_sales.db")

PLOT_DIR = os.path.join(DATA_DIR, "plots")
FINAL_PRED_CSV = os.path.join(DATA_DIR, "final_predictions_detailed.csv")
SCORING_CSV = os.path.join(DATA_DIR, "model_scoring_summary.csv")
CLEANED_CSV = os.path.join(DATA_DIR, "clean_sales_final.csv")

os.makedirs(PLOT_DIR, exist_ok=True)


# -----------------------
# UTIL: CSV loader with fallbacks
# -----------------------
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    except pd.errors.ParserError:
        return pd.read_csv(path, sep=";")


# -----------------------
# STEP 1 — Load raw CSVs & normalise headers
# -----------------------
print("1) Loading raw CSVs...")

raw_df = load_csv(RAW_CSV_PATH)
translations_df = load_csv(TRANSLATION_CSV_PATH)

raw_df.columns = raw_df.columns.str.title()
translations_df.columns = translations_df.columns.str.title()

if "Year_Month" not in raw_df.columns:
    raise KeyError("CSV must contain 'Year_Month' column.")

# optional backup of the raw data used for SQL
raw_df.to_csv(os.path.join(DATA_DIR, "raw_backup_before_sql.csv"), index=False)


# -----------------------
# STEP 2 — Create SQLite DB and load raw tables
# -----------------------
print("2) Creating SQLite DB and writing raw + translations tables...")

conn = sqlite3.connect(SQL_DB_PATH)
raw_df.to_sql("raw_sales", conn, if_exists="replace", index=False)
translations_df.to_sql("translations", conn, if_exists="replace", index=False)

with conn:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_make ON raw_sales(Make);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_body ON raw_sales(Body_Type);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_yearmonth ON raw_sales(Year_Month);")

print("Raw tables written into:", SQL_DB_PATH)


# -----------------------
# STEP 3 — Advanced SQL cleaning (ETL)
# -----------------------
print("3) Running advanced SQL ETL (dedupe, typing, normalisation)...")

cleaning_sql = """
-- 0) Helper: create a normalized makes table using translations where available
DROP TABLE IF EXISTS makes;
CREATE TABLE makes AS
SELECT DISTINCT
       COALESCE(t.Make_English, TRIM(rs.Make)) AS Make_Normalised
FROM raw_sales rs
LEFT JOIN translations t
  ON TRIM(rs.Make) = TRIM(t.Make);

-- Add IDs
DROP TABLE IF EXISTS makes_id;
CREATE TABLE makes_id AS
SELECT ROW_NUMBER() OVER (ORDER BY Make_Normalised) AS MakeID,
       Make_Normalised
FROM makes;

-- 1) Normalise Year_Month to ISO text (YYYY-MM-DD or YYYY-MM)
DROP TABLE IF EXISTS raw_parsed;
CREATE TABLE raw_parsed AS
SELECT
    *,
    TRIM(Make)      AS Make_raw,
    TRIM(Body_Type) AS Body_Type_raw,
    Year_Month      AS Year_Month_raw,
    CASE
      WHEN LENGTH(Year_Month) >= 7 
           AND substr(Year_Month,5,1) IN ('-','/') THEN
           replace(Year_Month, '/', '-')
      ELSE
           NULL
    END AS Year_Month_iso
FROM raw_sales;

-- 2) Remove rows with missing essential fields (Make, Body_Type, Units_Sold)
DROP TABLE IF EXISTS raw_filtered;
CREATE TABLE raw_filtered AS
SELECT *
FROM raw_parsed
WHERE COALESCE(Make_raw, '') <> ''
  AND COALESCE(Body_Type_raw, '') <> ''
  AND Units_Sold IS NOT NULL;

-- 3) Convert Units_Sold to numeric
DROP TABLE IF EXISTS raw_numeric;
CREATE TABLE raw_numeric AS
SELECT *,
       CASE
         WHEN TRIM(Units_Sold) = '' THEN NULL
         ELSE CAST(REPLACE(Units_Sold, ',', '') AS INTEGER)
       END AS Units_Sold_n
FROM raw_filtered;

-- 4) Remove rows where Units_Sold_n is null or <= 0
DROP TABLE IF EXISTS raw_positive;
CREATE TABLE raw_positive AS
SELECT *
FROM raw_numeric
WHERE Units_Sold_n IS NOT NULL AND Units_Sold_n > 0;

-- 5) Deduplicate by (Make_raw, Body_Type_raw, Year_Month_canonical)
DROP TABLE IF EXISTS raw_dedup_prep;
CREATE TABLE raw_dedup_prep AS
SELECT *,
       COALESCE(Year_Month_iso, Year_Month_raw) AS Year_Month_canonical
FROM raw_positive;

DROP TABLE IF EXISTS raw_dedup;
CREATE TABLE raw_dedup AS
SELECT *
FROM (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY Make_raw, Body_Type_raw, Year_Month_canonical
           ORDER BY Units_Sold_n DESC
         ) AS rn
  FROM raw_dedup_prep
)
WHERE rn = 1;

-- 6) Cast canonical date to ISO YYYY-MM-01
DROP TABLE IF EXISTS raw_dates;
CREATE TABLE raw_dates AS
SELECT *,
       CASE
         WHEN LENGTH(Year_Month_canonical) >= 7 
         THEN substr(Year_Month_canonical,1,7) || '-01'
         ELSE NULL
       END AS Year_Month_clean
FROM raw_dedup;

-- 7) Remove rows with invalid dates
DROP TABLE IF EXISTS raw_valid_dates;
CREATE TABLE raw_valid_dates AS
SELECT *
FROM raw_dates
WHERE Year_Month_clean IS NOT NULL;

-- 8) Join to normalized makes_id to get MakeID and canonical Make
DROP TABLE IF EXISTS clean_sales;
CREATE TABLE clean_sales AS
SELECT
    rd.*,
    mi.MakeID,
    mi.Make_Normalised AS Make_Normalised
FROM raw_valid_dates rd
LEFT JOIN makes_id mi
  ON mi.Make_Normalised = COALESCE(
       (SELECT Make_English FROM translations 
        WHERE TRIM(translations.Make)=TRIM(rd.Make_raw)),
       TRIM(rd.Make_raw)
     );

-- 9) Final clean: output only translated + cleaned columns
DROP TABLE IF EXISTS clean_sales_final;
CREATE TABLE clean_sales_final AS
SELECT
    mi.MakeID,
    mi.Make_Normalised AS Make,
    TRIM(rd.Body_Type_raw) AS Body_Type,
    date(rd.Year_Month_clean) AS Year_Month,
    rd.Units_Sold_n AS Units_Sold
FROM clean_sales rd
LEFT JOIN makes_id mi
    ON mi.MakeID = rd.MakeID;

CREATE INDEX IF NOT EXISTS idx_clean_make  ON clean_sales_final(Make);
CREATE INDEX IF NOT EXISTS idx_clean_body ON clean_sales_final(Body_Type);
CREATE INDEX IF NOT EXISTS idx_clean_date ON clean_sales_final(Year_Month);
"""

with conn:
    conn.executescript(cleaning_sql)

tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table';", conn
)
print("Tables in DB after cleaning:", tables["name"].tolist())


# -----------------------
# STEP 4 — Load cleaned table into pandas
# -----------------------
print("4) Loading cleaned data into pandas...")

clean_df = pd.read_sql_query(
    "SELECT * FROM clean_sales_final ORDER BY Make, Body_Type, Year_Month;", conn
)
clean_df["Year_Month"] = pd.to_datetime(clean_df["Year_Month"])

clean_df.to_csv(CLEANED_CSV, index=False)
print(f"Saved cleaned CSV to: {CLEANED_CSV}")


# -----------------------
# STEP 5 — Feature engineering
# -----------------------
print("5) Feature engineering...")

def add_time_features(df, ref_date=None):
    df = df.copy()
    if ref_date is None:
        ref_date = df["Year_Month"].min()
    df["Year"] = df["Year_Month"].dt.year
    df["Month"] = df["Year_Month"].dt.month
    df["Month_Since_Start"] = (
        (df["Year_Month"].dt.year - ref_date.year) * 12
        + (df["Year_Month"].dt.month - ref_date.month)
    )
    return df

groups = {}
for (make, body), grp in clean_df.groupby(["Make", "Body_Type"]):
    if grp["Year_Month"].nunique() < 16:
        continue
    groups[(make, body)] = grp.sort_values("Year_Month").reset_index(drop=True)

print(f"Prepared {len(groups)} groups for modelling.")


# -----------------------
# STEP 6 — Modelling per group
# -----------------------
print("6) Training models per group...")

models = {}
predictions_list = []

for (make, body), grp in groups.items():
    months = grp["Year_Month"].unique()
    n = len(months)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    train_m = months[:train_end]
    val_m = months[train_end:val_end]
    test_m = months[val_end:]

    train_df = grp[grp["Year_Month"].isin(train_m)].copy()
    val_df = grp[grp["Year_Month"].isin(val_m)].copy()
    test_df = grp[grp["Year_Month"].isin(test_m)].copy()

    ref_date = train_df["Year_Month"].min()
    train_df = add_time_features(train_df, ref_date)
    val_df = add_time_features(val_df, ref_date)
    test_df = add_time_features(test_df, ref_date)

    feature_cols = ["Year", "Month", "Month_Since_Start", "Make", "Body_Type"]
    target_col = "Units_Sold"

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Make", "Body_Type"],
            ),
            ("num", "passthrough", ["Year", "Month", "Month_Since_Start"]),
        ]
    )

    param_grid = [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3},
    ]

    best_pipeline = None
    best_mae = np.inf

    for params in param_grid:
        model = GradientBoostingRegressor(**params)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        try:
            pipeline.fit(train_df[feature_cols], train_df[target_col])
        except Exception as e:
            print(f"Model training error for {make}/{body}, params {params}: {e}")
            continue

        preds = pipeline.predict(val_df[feature_cols])
        mae = mean_absolute_error(val_df[target_col], preds)
        if mae < best_mae:
            best_mae = mae
            best_pipeline = pipeline

    if best_pipeline is None:
        continue

    models[(make, body)] = best_pipeline

    for split_name, df_split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        df_split = df_split.copy()
        df_split["Split"] = split_name
        df_split["Predicted_Units_Sold"] = best_pipeline.predict(df_split[feature_cols])
        df_split["Make"] = make
        df_split["Body_Type"] = body
        predictions_list.append(df_split)

if not predictions_list:
    raise ValueError("No predictions generated — check that groups have >=16 months.")

final_predictions = (
    pd.concat(predictions_list, ignore_index=True)
    .sort_values(["Make", "Body_Type", "Year_Month"])
    .reset_index(drop=True)
)

final_predictions.to_csv(FINAL_PRED_CSV, index=False)
print(f"Predictions saved to: {FINAL_PRED_CSV}")


# -----------------------
# STEP 7 — Evaluation & scoring
# -----------------------
print("7) Computing scoring metrics (MAE, RMSE)...")

metrics = []
for (make, body), grp in final_predictions.groupby(["Make", "Body_Type"]):
    val_test = grp[grp["Split"].isin(["Validation", "Test"])].copy()
    if len(val_test) == 0:
        continue
    y_true = val_test["Units_Sold"]
    y_pred = val_test["Predicted_Units_Sold"]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    metrics.append(
        {"Make": make, "Body_Type": body, "MAE": mae, "RMSE": rmse, "N": len(val_test)}
    )

metrics_df = pd.DataFrame(metrics).sort_values("MAE").reset_index(drop=True)
metrics_df.to_csv(SCORING_CSV, index=False)

print(f"Scoring saved to: {SCORING_CSV}")
print("Top groups by MAE:")
print(metrics_df.head(10))


# -----------------------
# STEP 8 — Plotting
# -----------------------
print("8) Generating plots...")

top_groups = metrics_df.head(5)[["Make", "Body_Type"]].values.tolist()

for make, body in top_groups:
    group_df = final_predictions[
        (final_predictions["Make"] == make)
        & (final_predictions["Body_Type"] == body)
    ].copy()
    if group_df.empty:
        continue

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(group_df["Year_Month"], group_df["Units_Sold"], label="Actual", linewidth=1.5)
    ax.plot(
        group_df["Year_Month"],
        group_df["Predicted_Units_Sold"],
        linestyle="--",
        label="Predicted",
    )

    for split_label, colour, alpha in [
        ("Train", "#eeeeee", 0.4),
        ("Validation", "#dddddd", 0.25),
        ("Test", "#cccccc", 0.2),
    ]:
        part = group_df[group_df["Split"] == split_label]
        if len(part):
            ax.fill_between(
                part["Year_Month"],
                part["Units_Sold"].min() - 1,
                part["Units_Sold"].max() + 1,
                facecolor=colour,
                alpha=alpha,
            )

    ax.set_title(f"Actual vs Predicted — {make} / {body}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    plt.tight_layout()

    safe_name = f"{make}_{body}".replace(" ", "_").replace("/", "_")
    save_path = os.path.join(PLOT_DIR, f"{safe_name}.png")
    fig.savefig(save_path, dpi=200)
    print(f"Saved plot: {save_path}")
    plt.show()


# -----------------------
# STEP 9 — SQL insights
# -----------------------
print("9) Running SQL analytics on cleaned table...")

q1 = """
SELECT Make, SUM(Units_Sold) AS Total_Sales
FROM clean_sales_final
GROUP BY Make
ORDER BY Total_Sales DESC
LIMIT 10;
"""
print("Top makes by total sales:")
print(pd.read_sql_query(q1, conn))

q2 = """
SELECT strftime('%Y-%m', Year_Month) AS YearMonth, SUM(Units_Sold) AS Total_Units
FROM clean_sales_final
GROUP BY YearMonth
ORDER BY YearMonth
LIMIT 12;
"""
print("\nRecent monthly sales trend (first 12 months):")
print(pd.read_sql_query(q2, conn))

q3 = """
WITH Monthly AS (
  SELECT Make, strftime('%Y-%m', Year_Month) AS ym, SUM(Units_Sold) AS monthly_units
  FROM clean_sales_final
  GROUP BY Make, ym
),
Growth AS (
  SELECT
    Make,
    ym,
    monthly_units,
    LAG(monthly_units) OVER (PARTITION BY Make ORDER BY ym) AS prev_units
  FROM Monthly
)
SELECT Make, ym, monthly_units, prev_units,
       ROUND(100.0 * (monthly_units - prev_units) / prev_units, 2) AS growth_pct
FROM Growth
WHERE prev_units IS NOT NULL
ORDER BY growth_pct DESC
LIMIT 10;
"""
print("\nFastest growing makes (MoM % growth):")
print(pd.read_sql_query(q3, conn))

conn.close()
print("Pipeline complete.")
