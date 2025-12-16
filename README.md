# Data Science in Production: MLOps and Software Engineering (Autumn 2025) Exam Project - Rawdockers

This repository contains the solution for the Data Science in Production: MLOps and Software Engineering Exam Project. The project creates a machine learning pipeline that processes user behavior data to identify potential new customers.

The solution has been refactored from a monolithic Python notebook into a standardized MLOps project structure, utilizing Dagger for pipeline orchestration and GitHub Actions for CI/CD automation.

## Project Structure

The project adheres to a standard Cookiecutter Data Science structure (ccds). We decomposed the original Jupyter Notebook monolith into the ccds directory and separating the orchestration logic into the `dagger` directory.

```text
itu-sdse-project/
├── .github/                # GitHub Actions workflows
├── ccds/                   # Main application code (Python)
│   ├── data/               # Data storage
│   ├── itu_sdse_project/   # Source code package
│   │   ├── pipeline.py     # Main training pipeline script
│   │   └── ...             # Feature engineering and data processing modules
│   ├── train.dockerfile    # Docker definition for the training environment
│   └── pyproject.toml      # Python dependencies
├── dagger/                 # Dagger orchestration logic (Go)
│   └── main.go             # Pipeline definition (Build -> Train -> Export)
└── model.pkl               # Generated model artifact (after running pipeline)
```

## How to Run the Code

This project uses Dagger to containerize and run the training pipeline. You can run the workflow locally or rely on the GitHub Actions automation.
Prerequisites
• Docker Engine (Running)
• Dagger CLI (If running locally)
• Git

Running the Dagger Pipeline Locally
To build the Docker container, pull data via DVC, train the model, and export the artifact locally, execute the following command from the root of the repository:

```text
dagger call train --source=./ccds --github-token=env:DVC_GITHUB_TOKEN --output=./model.pkl
```

Note: You must ensure DVC_GITHUB_TOKEN is set in your environment to allow DVC to pull data from the storage repository.

## GitHub Automation Workflow

The project includes a GitHub automation workflow located in .github/workflows/ which handles the CI/CD process.

## Workflow: Train Model
This workflow is triggered on pushes and pull requests to the main branch. It performs the following steps:
1. Environment Setup: Installs the Dagger CLI.
2. Pipeline Execution: Runs the Dagger train function, which:
    ◦ Builds the training container using ccds/train.dockerfile.
    ◦ Authenticates with Git using the provided secrets.
    ◦ Pulls the raw input data using DVC.
    ◦ Executes the Python pipeline (itu_sdse_project.pipeline).
3. Artifact Management: The trained model is exported and uploaded to GitHub Artifacts with the name model.
4. Validation: A dependent job downloads the model artifact and runs inference tests using the itu-sdse-project-model-validator action to ensure the model functions correctly.