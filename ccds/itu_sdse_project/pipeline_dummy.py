"""
Dummy pipeline that simulates the real pipeline.py
Trains a simple model on fake data and saves it as a pickle file.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path


def train():
    """Train a dummy model and save it."""
    # Generate fake data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Train simple model
    model = LogisticRegression()
    model.fit(X, y)

    # Save model
    output_path = Path(__file__).parent.parent / "models" / "model.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    train()
