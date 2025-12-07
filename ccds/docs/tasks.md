# Project Tasks - Notebook Refactoring

**Team Size:** 3 people  
**Goal:** Refactor monolithic notebook into modular Python project structure

---

## Phase 1: Project Setup & Infrastructure

### Task 1.1: Environment Setup
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Set up development environment and dependencies
- [ ] Create/update virtual environment with Python 3.11+
- [ ] Update `pyproject.toml` to support Python 3.11+
- [ ] Install package in editable mode (`pip install -e .`)
- [ ] Verify all team members can run the environment
- [ ] Document setup steps in README

### Task 1.2: Create Module Structure
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Create the folder structure for refactored code
- [ ] Create `itu_sdse_project/modeling/` directory
- [ ] Create empty Python files: `loaders.py`, `cleaners.py`, `transformers.py`, `splitters.py`, `train.py`, `evaluate.py`, `helpers.py`, `registry.py`
- [ ] Create `__init__.py` files in all directories
- [ ] Update imports in `__init__.py` files

---

## Phase 2: Data Processing Modules

### Task 2.1: Helper Functions (`helpers.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Extract utility functions from notebook
- [ ] Extract `describe_numeric_col()` function
- [ ] Extract `impute_missing_values()` function
- [ ] Extract `create_dummy_cols()` function
- [ ] Add docstrings and type hints
- [ ] Write unit tests for helper functions

### Task 2.2: Data Loaders (`loaders.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Implement data loading functionality
- [ ] Create `pull_dvc_data()` function (DVC pull with subprocess)
- [ ] Create `load_raw_data()` function (read CSV)
- [ ] Implement date filtering logic (min_date, max_date)
- [ ] Save date limits to JSON artifact
- [ ] Add logging with loguru
- [ ] Write unit tests

### Task 2.3: Data Cleaners (`cleaners.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Implement data cleaning functions
- [ ] Create `clean_data()` function (remove empty target, invalid data)
- [ ] Create `filter_by_source()` function (filter by signup/google/etc)
- [ ] Implement feature selection (drop irrelevant columns)
- [ ] Convert empty strings to NaN
- [ ] Add logging
- [ ] Write unit tests

### Task 2.4: Data Transformers (`transformers.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Implement feature transformation functions
- [ ] Create `create_categorical_columns()` function
- [ ] Create `separate_continuous_categorical()` function
- [ ] Create `remove_outliers()` function (Z-score clipping)
- [ ] Create `impute_missing_data()` function
- [ ] Create `scale_features()` function (MinMaxScaler)
- [ ] Create `save_scaler()` and `load_scaler()` functions
- [ ] Create `create_binned_features()` function (bin_source)
- [ ] Create `create_dummy_variables()` function
- [ ] Add logging
- [ ] Write unit tests

### Task 2.5: Data Splitters (`splitters.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Implement train/test splitting
- [ ] Create `split_train_test()` function
- [ ] Implement stratified splitting
- [ ] Save train/test datasets to CSV
- [ ] Add logging
- [ ] Write unit tests

---

## Phase 3: Model Training Modules

### Task 3.1: Model Training (`train.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Implement model training functions
- [ ] Create `train_xgboost()` function with RandomizedSearchCV
- [ ] Create `train_logistic_regression()` function with RandomizedSearchCV
- [ ] Implement MLflow experiment tracking
- [ ] Create `save_model()` functions for both models
- [ ] Save model artifacts (JSON for XGBoost, PKL for LR)
- [ ] Log parameters and metrics to MLflow
- [ ] Add logging
- [ ] Write unit tests

### Task 3.2: Model Evaluation (`evaluate.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Implement model evaluation functions
- [ ] Create `evaluate_model()` function (accuracy, confusion matrix, classification report)
- [ ] Create `compare_models()` function (compare f1-scores)
- [ ] Create `print_confusion_matrix()` function
- [ ] Create `print_classification_report()` function
- [ ] Save evaluation results to JSON
- [ ] Add logging
- [ ] Write unit tests

### Task 3.3: Model Prediction (`predict.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Implement prediction/inference functions
- [ ] Create `load_model()` function (from MLflow registry)
- [ ] Create `load_model_from_file()` function
- [ ] Create `load_scaler()` function
- [ ] Create `make_predictions()` function
- [ ] Save predictions to CSV
- [ ] Add logging
- [ ] Write unit tests

### Task 3.4: Model Registry (`registry.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Implement MLflow model registry functions
- [ ] Create `register_model()` function
- [ ] Create `get_production_model()` function
- [ ] Create `compare_models()` function (prod vs new)
- [ ] Create `transition_model_stage()` function (None → Staging → Production)
- [ ] Create `wait_until_ready()` helper function
- [ ] Add logging
- [ ] Write unit tests

---

## Phase 4: Pipeline Orchestration

### Task 4.1: Main Pipeline (`pipeline.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Create main orchestration pipeline
- [ ] Implement `train()` function (complete training workflow)
- [ ] Implement `predict()` function (complete inference workflow)
- [ ] Add step-by-step logging
- [ ] Handle errors gracefully
- [ ] Add command-line entry point
- [ ] Write integration tests

### Task 4.2: CCDS Integration (`dataset.py`, `features.py`)
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Integrate with CCDS boilerplate files
- [ ] Update `dataset.py` to call data processing modules
- [ ] Update `features.py` to call feature engineering modules
- [ ] Remove typer decorators (keep simple)
- [ ] Add proper imports
- [ ] Test CLI commands

---

## Phase 5: Testing & Documentation

### Task 5.1: Unit Tests
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Write comprehensive unit tests
- [ ] Create test files for each module in `tests/`
- [ ] Test helper functions
- [ ] Test data loading and cleaning
- [ ] Test transformations
- [ ] Test model training (mock heavy operations)
- [ ] Achieve >80% code coverage

### Task 5.2: Integration Tests
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Write end-to-end integration tests
- [ ] Test full training pipeline
- [ ] Test full inference pipeline
- [ ] Test artifact creation and loading
- [ ] Test MLflow integration

### Task 5.3: Documentation
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Document the refactored codebase
- [ ] Add docstrings to all functions (with type hints)
- [ ] Update README with usage instructions
- [ ] Document module structure
- [ ] Add code examples
- [ ] Document MLflow setup
- [ ] Document DVC usage

---

## Phase 6: Docker & CI/CD

### Task 6.1: Dockerization
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Create Docker container for the project
- [ ] Write Dockerfile
- [ ] Install dependencies in container
- [ ] Test training pipeline in Docker
- [ ] Test inference pipeline in Docker
- [ ] Document Docker usage

### Task 6.2: GitHub Actions
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Set up CI/CD workflows
- [ ] Create workflow for running tests
- [ ] Create workflow for training pipeline
- [ ] Create workflow for model deployment
- [ ] Add code quality checks (linting, formatting)
- [ ] Document workflow usage

---

## Phase 7: Cleanup & Validation

### Task 7.1: Code Quality
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Ensure code quality standards
- [ ] Remove commented-out code
- [ ] Follow PEP 8 style guide
- [ ] Add type hints to all functions
- [ ] Run linter (ruff, flake8, or black)
- [ ] Fix all linting errors

### Task 7.2: Validation
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Validate refactored code matches original
- [ ] Run original notebook and save outputs
- [ ] Run refactored pipeline and save outputs
- [ ] Compare model performance (accuracy, f1-score)
- [ ] Compare artifact files
- [ ] Document any differences

### Task 7.3: Final Review
**Owner:** [Assign]  
**Status:** ⏳ Not Started  
**Description:** Final team review before submission
- [ ] Code review by all team members
- [ ] Test all workflows end-to-end
- [ ] Review documentation completeness
- [ ] Prepare demo/presentation
- [ ] Final commit and tag release

---

## Notes for Team Distribution

**Suggested Assignment:**
- **Person 1:** Data Processing (Tasks 2.1-2.5)
- **Person 2:** Model Training & Evaluation (Tasks 3.1-3.4)
- **Person 3:** Pipeline, Testing & Docker (Tasks 4.1-4.2, 6.1-6.2)
- **Everyone:** Phase 5 & 7 (shared responsibility)

**Communication:**
- Use GitHub issues to track tasks
- Create separate branches for each major task
- Regular standups to sync progress
- Code reviews before merging to main
