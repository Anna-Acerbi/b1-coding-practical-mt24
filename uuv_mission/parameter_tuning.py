import numpy as np
from typing import Tuple, Dict
from .dynamic import Submarine, ClosedLoop, Mission
from .control import PDController

def find_optimal_parameters(
    mission: Mission,
    kp_range: Tuple[float, float, int],
    kd_range: Tuple[float, float, int],
    variance: float = 0.5
) -> Dict:
    """
    Find optimal PD parameters through grid search
    Returns dictionary with optimal parameters and their MSE
    """
    kp_values = np.linspace(kp_range[0], kp_range[1], kp_range[2])
    kd_values = np.linspace(kd_range[0], kd_range[1], kd_range[2])
    
    best_mse = float('inf')
    best_kp = 0
    best_kd = 0
    
    sub = Submarine()
    
    for kp in kp_values:
        for kd in kd_values:
            controller = PDController(KP=kp, KD=kd)
            closed_loop = ClosedLoop(sub, controller)
            trajectory = closed_loop.simulate_with_random_disturbances(mission, variance)
            mse = np.mean((mission.reference - trajectory.position[:, 1]) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_kp = kp
                best_kd = kd
                
    return {
        'KP': best_kp,
        'KD': best_kd,
        'mse': best_mse
    }

def grid_search_pd_parameters(
    mission: Mission,
    kp_range: Tuple[float, float, int] = (0.1, 1.0, 20),
    kd_range: Tuple[float, float, int] = (0.1, 1.0, 20),
    variance: float = 0.5
) -> Dict:
    """
    Perform grid search and return both optimal parameters and simulated trajectory
    """
    kp_values = np.linspace(kp_range[0], kp_range[1], kp_range[2])
    kd_values = np.linspace(kd_range[0], kd_range[1], kd_range[2])
    
    best_mse = float('inf')
    best_kp = 0
    best_kd = 0
    best_trajectory = None
    
    sub = Submarine()
    
    for kp in kp_values:
        for kd in kd_values:
            controller = PDController(KP=kp, KD=kd)
            closed_loop = ClosedLoop(sub, controller)
            
            # Run simulation
            trajectory = closed_loop.simulate_with_random_disturbances(mission, variance)
            
            # Calculate MSE
            mse = np.mean((mission.get_reference_trajectory() - 
                          trajectory.get_actual_trajectory()) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_kp = kp
                best_kd = kd
                best_trajectory = trajectory
                
    return {
        'parameters': {'KP': best_kp, 'KD': best_kd},
        'performance': {'mse': best_mse},
        'trajectory': best_trajectory
    }

def parameter_sweep_analysis(mission: Mission, **kwargs) -> Dict:
    """
    Perform parameter sweep and return complete analysis including trajectory
    """
    return grid_search_pd_parameters(mission, **kwargs)
