import warnings
from pathlib import Path
from pprint import pprint
import datetime
import json

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
import joblib

from itu_sdse_project.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from itu_sdse_project.modeling.loaders import load_raw_data, pull_dvc_data


warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: "%.3f" % x)


def describe_numeric_col(x):
    """
    describe_numeric_col: Calculates various descriptive stats for a numeric column in a dataframe.
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """
    impute_missing_values: Imputes the mean/median for numeric columns or the mode for other types.
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x


def clean_raw_data(
    df: pd.DataFrame,
    min_date: str = "2024-01-01",
    max_date: str | None = "2024-01-31",
    artifacts_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Full data cleaning pipeline from notebook.

    Steps:
      1. Date filtering on `date_part` (min_date / max_date)
      2. Feature selection (drop unused columns)
      3. Target & ID cleaning (lead_indicator, lead_id, customer_code)
      4. Filter source == 'signup'
      5. Cast selected columns to categorical (object)
      6. Split into continuous / categorical
      7. Clip outliers (±2 std) and save outlier summary
      8. Impute missing values (num + cat) and save cat impute table
      9. Fit MinMaxScaler on continuous vars, save scaler, and transform
     10. Combine cat + cont back together
     11. Save:
         - date_limits.json
         - outlier_summary.csv
         - cat_missing_impute.csv
         - columns_drift.json
         - training_data.csv
         - train_data_gold.csv
         - scaler.pkl
     12. Create binned source column `bin_source`

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe to clean.
    min_date : str
        Minimum date for `date_part` filtering (YYYY-MM-DD).
    max_date : str | None
        Maximum date for `date_part` filtering (YYYY-MM-DD). If None or empty, uses today.
    artifacts_dir : Path | None
        Directory where artifacts are saved. Defaults to ./artifacts under PROCESSED_DATA_DIR.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe, ready for modeling.
    """
    logger.info("Starting full cleaning pipeline...")
    data = df.copy()

    # Artifacts directory
    if artifacts_dir is None:
        artifacts_dir = PROCESSED_DATA_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Artifacts will be saved in: {artifacts_dir}")

    
    # 1. Date filtering on date_part
    if not max_date:
        max_date_parsed = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date_parsed = pd.to_datetime(max_date).date()

    min_date_parsed = pd.to_datetime(min_date).date()

    if "date_part" in data.columns:
        logger.info(f"Filtering date_part between {min_date_parsed} and {max_date_parsed}")
        data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
        data = data[
            (data["date_part"] >= min_date_parsed)
            & (data["date_part"] <= max_date_parsed)
        ]

        min_date_actual = data["date_part"].min()
        max_date_actual = data["date_part"].max()
        date_limits = {
            "min_date": str(min_date_actual),
            "max_date": str(max_date_actual),
        }

        with open(artifacts_dir / "date_limits.json", "w") as f:
            json.dump(date_limits, f)

        logger.info(
            f"Date filter applied. Rows remaining: {len(data)}, "
            f"min_date={min_date_actual}, max_date={max_date_actual}"
        )
    else:
        logger.warning("Column 'date_part' not found; skipping date filtering.")

    # 2. Feature selection / dropping columns
    drop_cols_1 = [
        "is_active",
        "marketing_consent",
        "first_booking",
        "existing_customer",
        "last_seen",
    ]
    drop_cols_2 = [
        "domain",
        "country",
        "visited_learn_more_before_booking",
        "visited_faq",
    ]

    for cols in (drop_cols_1, drop_cols_2):
        existing = [c for c in cols if c in data.columns]
        if existing:
            logger.info(f"Dropping columns: {existing}")
            data = data.drop(existing, axis=1)


    # 3. Data cleaning: target & IDs
    for col in ["lead_indicator", "lead_id", "customer_code"]:
        if col in data.columns:
            data[col].replace("", np.nan, inplace=True)

    for col in ["lead_indicator", "lead_id"]:
        if col in data.columns:
            before = len(data)
            data = data.dropna(axis=0, subset=[col])
            logger.info(f"Dropped {before - len(data)} rows with empty {col}")

    if "source" in data.columns:
        before = len(data)
        data = data[data["source"] == "signup"]
        logger.info(f"Filtered to source == 'signup'. Dropped {before - len(data)} rows.")

    if "lead_indicator" in data.columns:
        result = data["lead_indicator"].value_counts(normalize=True)
        logger.info("Target value counter (proportions):")
        for val, n in zip(result.index, result):
            logger.info(f"  {val}: {n:.3f}")


    # 4. Cast selected columns to categorical (object)

    cat_cols_force_object = [
        "lead_id",
        "lead_indicator",
        "customer_group",
        "onboarding",
        "source",
        "customer_code",
    ]

    for col in cat_cols_force_object:
        if col in data.columns:
            data[col] = data[col].astype("object")
            logger.info(f"Changed {col} to object type")


    # 5. Separate cont / cat variables
    cont_vars = data.loc[:, (data.dtypes == "float64") | (data.dtypes == "int64")]
    cat_vars = data.loc[:, data.dtypes == "object"]

    logger.info("Continuous columns:")
    pprint(list(cont_vars.columns), indent=4)

    logger.info("Categorical columns:")
    pprint(list(cat_vars.columns), indent=4)

    # 6. Outliers (clip at mean ± 2*std) + summary
    if not cont_vars.empty:
        logger.info("Clipping outliers at mean ± 2*std for continuous variables.")
        cont_vars = cont_vars.apply(
            lambda x: x.clip(lower=x.mean() - 2 * x.std(), upper=x.mean() + 2 * x.std())
        )

        outlier_summary = cont_vars.apply(describe_numeric_col).T
        outlier_summary.to_csv(artifacts_dir / "outlier_summary.csv")
        logger.info("Saved outlier_summary.csv")
    else:
        logger.warning("No continuous variables found; skipping outlier clipping.")

    
    # 7. Missing data imputation
    if not cat_vars.empty:
        cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
        cat_missing_impute.to_csv(artifacts_dir / "cat_missing_impute.csv", index=False)
        logger.info("Saved cat_missing_impute.csv")

    # Continuous vars
    if not cont_vars.empty:
        logger.info("Imputing missing values for continuous variables.")
        cont_vars = cont_vars.apply(impute_missing_values)

    # Categorical vars
    if "customer_code" in cat_vars.columns:
        cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"

    if not cat_vars.empty:
        logger.info("Imputing missing values for categorical variables.")
        cat_vars = cat_vars.apply(impute_missing_values)

    # 8. Data standardisation (MinMaxScaler)
    if not cont_vars.empty:
        logger.info("Fitting MinMaxScaler on continuous variables.")
        scaler = MinMaxScaler()
        scaler.fit(cont_vars)

        scaler_path = artifacts_dir / "scaler.pkl"
        joblib.dump(value=scaler, filename=scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        cont_vars = pd.DataFrame(
            scaler.transform(cont_vars),
            columns=cont_vars.columns,
            index=cont_vars.index,
        )
    
    # 9. Combine categorical + continuous
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)

    logger.info(
        f"Data cleansed and combined. Rows: {len(data)}, Columns: {len(data.columns)}"
    )

    # 10. Data drift artifact: column list + training_data
    data_columns = list(data.columns)
    with open(artifacts_dir / "columns_drift.json", "w") as f:
        json.dump(data_columns, f)
    logger.info("Saved columns_drift.json")

    training_data_path = artifacts_dir / "training_data.csv"
    data.to_csv(training_data_path, index=False)
    logger.info(f"Saved training_data.csv to {training_data_path}")

    # 11. Binning source column
    # -----------------------------
    if "source" in data.columns:
        logger.info("Creating binned source column 'bin_source'.")
        data["bin_source"] = data["source"]

        values_list = ["li", "organic", "signup", "fb"]
        data.loc[~data["source"].isin(values_list), "bin_source"] = "Others"

        mapping = {
            "li": "socials",
            "fb": "socials",
            "organic": "group1",
            "signup": "group1",
            "Others": "Others",
        }

        data["bin_source"] = data["bin_source"].map(mapping)

    # Final gold dataset
    gold_path = artifacts_dir / "train_data_gold.csv"
    data.to_csv(gold_path, index=False)
    logger.info(f"Saved train_data_gold.csv to {gold_path}")

    logger.success("Full cleaning pipeline completed.")
    return data


def run_cleaning_pipeline() -> pd.DataFrame:
    """
    Convenience wrapper:
      - dvc pull
      - load RAW_DATA_DIR/raw_data.csv
      - run clean_raw_data
    """
    logger.info("Running cleaning pipeline from cleaners.run_cleaning_pipeline()")
    pull_dvc_data()
    raw_path = RAW_DATA_DIR / "raw_data.csv"
    df_raw = load_raw_data(raw_path)
    return clean_raw_data(df_raw)
    

if __name__ == "__main__":
    # Allows: python -m itu_sdse_project.modeling.cleaners
    run_cleaning_pipeline()
