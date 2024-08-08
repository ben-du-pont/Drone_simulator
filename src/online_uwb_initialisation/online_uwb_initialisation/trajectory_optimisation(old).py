import numpy as np
from scipy.optimize import minimize
from itertools import repeat

from drone_uwb_simulator.drone_dynamics import Waypoint, Trajectory

class NonLinearTrajectoryOptimization:

    def __init__(self, anchor_positions = [[]], waypoints = [[]], weights = [0.5, 0.3, 0.1]):
        self.anchor_positions = anchor_positions
        self.waypoints = waypoints
        self.weights = weights
    
    def update_anchor_positions(self, anchor_positions):
        self.anchor_positions = anchor_positions

    def update_waypoints(self, waypoints):
        self.waypoints = waypoints

    def update_weights(self, weights):
        self.weights = weights

    def objective_function(self, trajectory, anchor_positions, waypoints, weights):
        # Compute angular spread component
        angular_spread = np.var(anchor_positions - trajectory[:, np.newaxis], axis=0).sum()
        
        # angular_spread_penalty = max(0, 100 - angular_spread)

        # Compute distance spread component
        distance_spread = np.var(np.linalg.norm(trajectory[:, np.newaxis] - anchor_positions, axis=0))
        
        # Compute deviation from initial trajectory component
        deviation = np.exp(np.linalg.norm(trajectory - waypoints))
        
        # Combine components using weights
        objective = -weights[0] * angular_spread - weights[1] * distance_spread + weights[2] * deviation # + 1e3* angular_spread_penalty
        
        return objective



    def optimize_trajectory(self, initial_trajectory, anchor_positions, waypoints, weights):
        # Define optimization bounds
        # Define optimization bounds for each dimension of each waypoint separately
        bounds_per_dimension = [(min(initial_trajectory[:, i]), max(initial_trajectory[:, i])) for i in range(waypoints.shape[1])]
        
        # Repeat bounds for each dimension of each waypoint
        bounds = bounds_per_dimension * len(waypoints)


        # Define optimization function
        optimization_func = lambda trajectory: self.objective_function(trajectory.reshape(-1, 3), anchor_positions, waypoints, weights)
        
        # Perform optimization
        result = minimize(optimization_func, initial_trajectory.flatten(), bounds=None)
        
        # Extract optimized trajectory
        optimized_trajectory = result.x.reshape(-1, 3)
        
        return optimized_trajectory
    

    def run_optimization(self):

        # Initial trajectory waypoints
        initial_waypoints = np.array(self.waypoints)

        # Rough estimates of anchor positions
        anchor_positions = np.array(self.anchor_positions)

        # Initial guess for the new trajectory (e.g., same as the initial trajectory)
        initial_trajectory_guess = initial_waypoints.copy()

        # Optimize the trajectory
        optimized_trajectory = self.optimize_trajectory(initial_trajectory_guess, anchor_positions, initial_waypoints, self.weights)

        return optimized_trajectory
    

    
    def create_optimised_trajectory(self, drone_position ,optimised_waypoints, idx_cut, drone_initial_trajectory):
        waypoints = []
        for i in range(idx_cut):
            waypoints.append(Waypoint(drone_initial_trajectory[i][0], drone_initial_trajectory[i][1], drone_initial_trajectory[i][2]))

        waypoints.append(Waypoint(drone_position[0], drone_position[1], drone_position[2]))
        for waypoint in optimised_waypoints:
            waypoints.append(Waypoint(waypoint[0], waypoint[1], waypoint[2]))

        drone_trajectory = Trajectory(speed=5, dt=0.05)
        drone_trajectory.construct_trajectory_linear(waypoints)

        return drone_trajectory.spline_x, drone_trajectory.spline_y, drone_trajectory.spline_z