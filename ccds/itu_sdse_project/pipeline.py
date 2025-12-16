"""Main pipeline orchestrator for the ML workflow."""

from loguru import logger
from pathlib import Path
import joblib
import json
import datetime
import os
import shutil

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

from itu_sdse_project.config import MODELS_DIR

try:
    import mlflow
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint
from mlflow.tracking import MlflowClient
import time

from itu_sdse_project.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from itu_sdse_project.modeling.loaders import load_raw_data, pull_dvc_data
from itu_sdse_project.modeling.cleaners import clean_raw_data


def train():
    """
    Training pipeline:
    - Load data 
    - Process data
    - Train models
    - Evaluate and select best model
    - Register model
    """

    logger.info("=" * 50)
    logger.info("Starting TRAINING pipeline...")
    logger.info("=" * 50)
    
    #1 Load & Clean data
    logger.info("Step 1: Loading data...")
    pull_dvc_data()
    df = load_raw_data(RAW_DATA_DIR / "raw_data.csv")

    logger.info("Step 2: Cleaning data...")
    df_cleaned = clean_raw_data(df)

    #2 Load gold dataset produced by cleaner
    artifacts_dir = PROCESSED_DATA_DIR / "artifacts"
    gold_path = artifacts_dir / "train_data_gold.csv"
    if not gold_path.exists():
        logger.warning(f"Gold dataset not found at {gold_path}, using cleaned dataframe in-memory")
        data = df_cleaned.copy()
    else:
        logger.info(f"Loading gold dataset from {gold_path}")
        data = pd.read_csv(gold_path)

    #3 Feature engineering / dummies
    logger.info("Step 3: Feature engineering (dummies / types)...")

    #Drop columns not used for training if present
    for c in ["lead_id", "customer_code", "date_part"]:
        if c in data.columns:
            data = data.drop(c, axis=1)

    cat_cols = [c for c in ["customer_group", "onboarding", "bin_source", "source"] if c in data.columns]
    cat_df = data[cat_cols].copy() if cat_cols else pd.DataFrame()
    other_df = data.drop(cat_cols, axis=1) if cat_cols else data.copy()

    def create_dummy_cols(df, col):
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        return df

    if not cat_df.empty:
        for col in list(cat_df.columns):
            cat_df[col] = cat_df[col].astype("category")
            cat_df = create_dummy_cols(cat_df, col)

    data_prepared = pd.concat([other_df, cat_df], axis=1)

    #Ensure floats
    for col in data_prepared.columns:
        try:
            data_prepared[col] = data_prepared[col].astype("float64")
        except Exception:
            # leave as-is (e.g., target column)
            pass

    #4 Split
    logger.info("Step 4: Splitting train/test...")
    if "lead_indicator" not in data_prepared.columns:
        raise RuntimeError("Target column 'lead_indicator' not found in prepared data")

    y = data_prepared["lead_indicator"]
    X = data_prepared.drop(["lead_indicator"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )

    #Save processed splits
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(artifacts_dir / "X_train.csv", index=False)
    X_test.to_csv(artifacts_dir / "X_test.csv", index=False)
    y_train.to_csv(artifacts_dir / "y_train.csv", index=False)
    y_test.to_csv(artifacts_dir / "y_test.csv", index=False)

    #5 Train candidate models
    logger.info("Step 5: Training candidate models...")
    model_results: dict = {}
    # Keep track of which models being logged to MLflow: {model_path: {run_id, artifact_path}}
    logged_models: dict[str, dict] = {}

    #Experiment name for MLflow (shared with all models)
    experiment_name = datetime.datetime.now().strftime("%Y_%B_%d")
    experiment_id = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        except Exception:
            logger.warning("Failed to set MLflow experiment; continuing without MLflow experiment id")

    #5A XGBoost randomized search
    try:
        xgb = XGBRFClassifier(random_state=42, use_label_encoder=False)
        params_xgb = {
            "learning_rate": uniform(1e-2, 3e-1),
            "min_split_loss": uniform(0, 10),
            "max_depth": randint(3, 10),
            "subsample": uniform(0, 1),
            "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
            "eval_metric": ["aucpr", "error"],
        }

        xgb_search = RandomizedSearchCV(xgb, param_distributions=params_xgb, n_iter=6, cv=3, n_jobs=-1, verbose=1)
        xgb_search.fit(X_train, y_train)
        y_pred_train = xgb_search.predict(X_train)
        y_pred_test = xgb_search.predict(X_test)

        xgb_path = artifacts_dir / "lead_model_xgboost.json"
        xgb_search.best_estimator_.save_model(str(xgb_path))

        #If MLflow is available, attempt to log this model and metrics so it can be registered later
        if MLFLOW_AVAILABLE:
            try:
                import importlib
                mlflow_local = importlib.import_module("mlflow")
                mlflow_sklearn = importlib.import_module("mlflow.sklearn")
                mlflow_local.set_experiment(experiment_name)
                exp = mlflow_local.get_experiment_by_name(experiment_name)
                exp_id = exp.experiment_id if exp is not None else None
                with mlflow_local.start_run(experiment_id=exp_id) as run:
                    try:
                        mlflow_sklearn.log_model(xgb_search.best_estimator_, artifact_path="model")
                        mlflow_local.log_metric("f1_score", f1_score(y_test, y_pred_test))
                        logged_models[str(xgb_path)] = {"run_id": run.info.run_id, "artifact_path": "model"}
                    except Exception:
                        logger.warning("Failed to log XGBoost to MLflow; continuing without MLflow log.")
            except Exception:
                logger.warning("MLflow not available at logging time for XGBoost; skipping MLflow logging")

        model_results[str(xgb_path)] = classification_report(y_test, y_pred_test, output_dict=True)
        logger.info(f"XGBoost training complete. Saved model to {xgb_path}")
    except Exception as exc:
        logger.error(f"XGBoost training failed: {exc}")

    #5B Logistic Regression with RandomizedSearchCV + mlflow logging
    try:
        if MLFLOW_AVAILABLE:
            try:
                mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
            except Exception:
                logger.warning("Failed to enable mlflow.sklearn.autolog")

        lr = LogisticRegression(max_iter=1000)
        params_lr = {
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "penalty": ["none", "l1", "l2", "elasticnet"],
            "C": [100, 10, 1.0, 0.1, 0.01],
        }

        lr_search = RandomizedSearchCV(lr, param_distributions=params_lr, n_iter=6, cv=3, verbose=1)
        lr_search.fit(X_train, y_train)

        y_pred_train = lr_search.predict(X_train)
        y_pred_test = lr_search.predict(X_test)

        lr_path = artifacts_dir / "lead_model_lr.pkl"
        joblib.dump(value=lr_search.best_estimator_, filename=str(lr_path))

        #Optional mlflow logging
        if MLFLOW_AVAILABLE:
            try:
                import importlib
                mlflow_local = importlib.import_module("mlflow")
                mlflow_pyfunc = importlib.import_module("mlflow.pyfunc")
                exp_id = experiment_id
                with mlflow_local.start_run(experiment_id=exp_id) as run:
                    from sklearn.metrics import f1_score as _f1

                    try:
                        mlflow_local.log_metric("f1_score", _f1(y_test, y_pred_test))
                        mlflow_local.log_artifacts(str(artifacts_dir), artifact_path="model")
                        mlflow_local.log_param("data_version", "00000")
                    except Exception:
                        logger.warning("Failed to log basic LR metrics/artifacts to MLflow")

                    #Log python model wrapper for probability predictions
                    class LRWrapper(mlflow_pyfunc.PythonModel):
                        def __init__(self, model):
                            self.model = model

                        def predict(self, context, model_input):
                            return self.model.predict_proba(model_input)[:, 1]

                    #Log pyfunc model under artifact path 'model'
                    try:
                        mlflow_pyfunc.log_model("model", python_model=LRWrapper(lr_search.best_estimator_), registered_model_name=None)
                        logged_models[str(lr_path)] = {"run_id": run.info.run_id, "artifact_path": "model"}
                    except Exception:
                        logger.warning("Failed to log LR as pyfunc model; continuing without pyfunc log.")
            except Exception:
                logger.warning("MLflow not available at logging time for LR; skipping MLflow logging")

        model_results[str(lr_path)] = classification_report(y_test, y_pred_test, output_dict=True)
        logger.info(f"Logistic Regression training complete. Saved model to {lr_path}")
    except Exception as exc:
        logger.error(f"Logistic regression training failed: {exc}")

    #6 Save results & select best model
    logger.info("Step 6: Saving model results and selecting best model...")
    results_path = artifacts_dir / "model_results.json"
    with open(results_path, "w+") as f:
        json.dump(model_results, f)

    cols_path = artifacts_dir / "columns_list.json"
    with open(cols_path, "w+") as f:
        json.dump({"column_names": list(X_train.columns)}, f)

    #Determine best model by weighted avg f1-score
    best_model_path = None
    best_f1 = -1.0
    for model_path, report in model_results.items():
        f1w = report.get("weighted avg", {}).get("f1-score")
        if f1w is None:
            continue
        if f1w > best_f1:
            best_f1 = f1w
            best_model_path = model_path

    logger.info(f"Best model: {best_model_path} with weighted f1 {best_f1}")

    #Optionally register with MLflow if available and an mlflow run exists
    if MLFLOW_AVAILABLE and best_model_path is not None:
        try:
            model_name = "lead_model"
            client = MlflowClient()

            #If model logged during training, retrieve run_id and artifact_path
            info = logged_models.get(best_model_path)
            if info:
                model_uri = f"runs:/{info['run_id']}/{info['artifact_path']}"
                logger.info(f"Registering model from {model_uri} to MLflow model registry as '{model_name}'")
                model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

                #Wait until the model version is READY
                def wait_until_ready(name, version, timeout=60):
                    start = time.time()
                    while time.time() - start < timeout:
                        mv = client.get_model_version(name=name, version=version)
                        status = mv.status
                        if status == "READY":
                            return True
                        time.sleep(1)
                    return False

                version = model_details.version
                ready = wait_until_ready(model_details.name, version)
                if ready:
                    #Transition to Production, archive other versions
                    client.transition_model_version_stage(
                        name=model_details.name,
                        version=version,
                        stage="Production",
                        archive_existing_versions=True,
                    )
                    logger.info(f"Model registered and transitioned to Production: {model_details.name} v{version}")
                else:
                    logger.warning("Model registered but did not reach READY state within timeout")
            else:
                logger.info("Best model file was not logged to MLflow during training; skipping automatic registry.")
        except Exception as exc:
            logger.error(f"Model registration failed: {exc}")

    logger.success("Training pipeline completed.")

    #Copy best model to models/model.pkl for the validator
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_model_path = MODELS_DIR / "model.pkl"
    
    if best_model_path:
        if best_model_path.endswith(".pkl"):
            shutil.copy(best_model_path, final_model_path)
        else:
            #For XGBoost JSON, load and re-save as pickle
            best_xgb = XGBRFClassifier()
            best_xgb.load_model(best_model_path)
            joblib.dump(best_xgb, final_model_path)
        logger.info(f"Copied best model to {final_model_path}")

    logger.success("Training pipeline completed.")


#Prediction and model loading utilities.
# Unutilised code for loading a pre-built model and scaler for inference.
"""
def load_scaler(scaler_path: Path):
    #Load saved scaler from disk.
    logger.info(f"Loading scaler from {scaler_path}")
    #Use joblib.load() to load scaler
    pass

def load_model(model_name: str = "lead_model", stage: str = "Production"):
    #Load model from MLflow registry.
    logger.info(f"Loading model: {model_name} (stage: {stage})")
    #Use mlflow.pyfunc.load_model() or load from file
    pass

def load_model_from_file(model_path: Path):
    #Load model from saved file.
    logger.info(f"Loading model from {model_path}")
    #Load XGBoost or pickle model from file
    pass
"""


if __name__ == "__main__":
    train()