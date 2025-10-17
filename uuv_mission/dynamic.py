from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from .terrain import generate_reference_and_limits

class Submarine:
    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

@dataclass
class Mission:
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        """
        Load mission data from a CSV file. Expected columns (case-insensitive):
        'reference', 'cave_height', 'cave_depth'.
        If these column names are not present, the method falls back to using
        the first three columns in the file in that order.
        """
        import csv
        import numpy as np

        ref_list = []
        h_list = []
        d_list = []

        with open(file_name, newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f"CSV file '{file_name}' is empty")

            lower_header = [h.strip().lower() for h in header]

            # Attempt to find named columns
            try:
                idx_ref = lower_header.index('reference')
                idx_h = lower_header.index('cave_height')
                idx_d = lower_header.index('cave_depth')

                for row in reader:
                    if not row:
                        continue
                    # guard against short rows
                    if max(idx_ref, idx_h, idx_d) >= len(row):
                        raise ValueError("Row in CSV has fewer columns than expected based on header")
                    ref_list.append(float(row[idx_ref]))
                    h_list.append(float(row[idx_h]))
                    d_list.append(float(row[idx_d]))
            except ValueError:
                # Fallback: assume first three columns are reference, cave_height, cave_depth
                # Reset reader to start after header
                f.seek(0)
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if not row:
                        continue
                    if len(row) < 3:
                        raise ValueError("CSV fallback expects at least 3 columns per row")
                    ref_list.append(float(row[0]))
                    h_list.append(float(row[1]))
                    d_list.append(float(row[2]))

        # Convert to numpy arrays and validate lengths
        reference = np.array(ref_list, dtype=float)
        cave_height = np.array(h_list, dtype=float)
        cave_depth = np.array(d_list, dtype=float)

        if not (len(reference) == len(cave_height) == len(cave_depth)):
            raise ValueError("Loaded columns have mismatched lengths")

        return cls(reference, cave_height, cave_depth)


class ClosedLoop:
    def __init__(self, plant: Submarine, controller):
        self.plant = plant
        self.controller = controller

    def simulate(self,  mission: Mission, disturbances: np.ndarray) -> Trajectory:

        T = len(mission.reference)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        actions = np.zeros(T)
        self.plant.reset_state()

        for t in range(T):
            positions[t] = self.plant.get_position()
            observation_t = self.plant.get_depth()
            # Call your controller here
            self.plant.transition(actions[t], disturbances[t])

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)
