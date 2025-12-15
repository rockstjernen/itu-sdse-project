FROM python:3.11-slim

WORKDIR /app

# Install git (required for DVC)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package
COPY itu_sdse_project/ ./itu_sdse_project/

# Copy DVC config and data pointers
COPY .dvc/ ./.dvc/
COPY data/ ./data/

# Create models directory
RUN mkdir -p models

# Run the pipeline
CMD ["python", "-m", "itu_sdse_project.pipeline"]
