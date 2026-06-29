"""
Entry point for the live dashboard.

    Aircraft Simulator -> Radar Simulator -> Data Logger -> Feature Extraction
        -> Isolation Forest -> Threat Engine -> Dashboard

Run:
    python main.py

Requires a trained model. If you haven't run the training pipeline yet:
    python generate_training_data.py
    python train_model.py
"""

from dashboard import run_dashboard

if __name__ == "__main__":
    run_dashboard()
