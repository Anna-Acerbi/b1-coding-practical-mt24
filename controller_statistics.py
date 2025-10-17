import numpy as np

def calculate_mse(reference_trajectory, actual_trajectory):
    """Calculate Mean Squared Error between reference and actual trajectories."""
    return np.mean((reference_trajectory - actual_trajectory) ** 2)

def report_statistics(initial_params, final_params, mse):
    """Report the initial and final parameters along with the MSE."""
    print(f"Initial Parameters: KP={initial_params[0]}, KD={initial_params[1]}")
    print(f"Final Parameters: KP={final_params[0]}, KD={final_params[1]}")
    print(f"Mean Squared Error: {mse:.4f}")
