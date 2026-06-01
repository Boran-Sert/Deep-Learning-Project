"""
Main entry point for the Time Series Anomaly Detection project.
Executes full experiments for SKAB and BATADAL datasets with all phases.
"""

import os
import random
import numpy as np
import torch

from src.core.config_manager import ConfigurationManager
from src.orchestration.orchestrator import ExperimentOrchestrator


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """Main function to run experiments for all datasets."""
    print("=" * 70)
    print("TIME SERIES ANOMALY DETECTION - MAIN EXPERIMENT RUNNER")
    print("=" * 70)
    
    # Load configuration
    config = ConfigurationManager()
    print("\nConfiguration loaded successfully.")
    
    # Get random seeds from config
    seeds = config.get("experiment.random_seeds", [42, 123, 2026, 7, 999])
    print(f"Random seeds configured: {seeds}")
    
    # Get datasets to run
    datasets = ["skab", "batadal"]
    print(f"Datasets to process: {datasets}")
    
    # Run experiments for each dataset
    for dataset in datasets:
        print("\n" + "=" * 70)
        print(f"STARTING EXPERIMENT: {dataset.upper()}")
        print("=" * 70)
        
        # Set seed for reproducibility
        set_seed(42)  # Use first seed for initialization
        
        # Create orchestrator and run experiment
        orchestrator = ExperimentOrchestrator()
        orchestrator.run_experiment(dataset)
        
        print(f"\nExperiment for {dataset.upper()} completed successfully!")
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated reports and visualizations are saved in the 'reports/' directory.")
    print("Check 'reports/experiment_report.json' for summary statistics.")
    print("Check 'reports/statistical_comparison.json' for model comparison results.")
    print("Check 'reports/figures/' for visualization plots.")


if __name__ == "__main__":
    main()
