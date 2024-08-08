import numpy as np
from scipy.optimize import minimize
from itertools import repeat
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_uwb_simulator.drone_dynamics import Waypoint, Trajectory




class TrajectoryOptimization:


    def __init__(self):
        self.fim_noise_variance = 0.2  # Noise variance for the FIM calculation
        


    def calculate_fim(self, anchor_position, measurements):
        """
        Calculate the Fisher Information Matrix (FIM) for the given anchor positions and trajectory.
        """
        x, y, z = anchor_position  # Define the target coordinates
        FIM = np.zeros((3, 3))  # Initialize the FIM matrix
        
        for measurement in measurements:
            x_i, y_i, z_i = measurement
            d_i = np.sqrt((x - x_i)**2 + (y - y_i)**2 + (z - z_i)**2)
            
            if d_i == 0:
                continue  # Avoid division by zero

            # Jacobian of the distance with respect to the anchor position
            jacobian = (np.array([x_i, y_i, z_i]) - np.array([x, y, z])) / d_i

            # Update FIM
            FIM += (1 / (self.fim_noise_variance * (1 + d_i**2))) * np.outer(jacobian, jacobian)
        fim_det = np.linalg.det(FIM)

        return 1/fim_det
    
    def compute_GDOP(self, target_coords, measurements):
        """Compute the Geometric Dilution of Precision (GDOP) given a set of measurements corresponding to drone positions in space"""
        
        if len(measurements) < 4:
            return float('inf') # Not enough points to calculate GDOP
    
        x,y,z = target_coords
        A = []
        for measurement in measurements:
            x_i, y_i, z_i = measurement
            R = np.linalg.norm([x_i-x, y_i-y, z_i-z])
            A.append([(x_i-x)/R, (y_i-y)/R, (z_i-z)/R, 1])

        A = np.array(A)

        try:
            inv_at_a = np.linalg.inv(A.T @ A)
            gdop = np.sqrt(np.trace(inv_at_a))
            if gdop is not None:
                return gdop
            else:
                return float('inf')
        except np.linalg.LinAlgError:
            return float('inf')  # Matrix is singular, cannot compute GDOP
        
    def evaluate_gdop(self, new_measurements, anchor_position, previous_measurements):
        """
        Evaluate the GDOP using the previous and new measurements.
        """
        # Calculate the GDOP using the previous and new measurements
        new_measurements = np.array(new_measurements).reshape((-1, 3))
        gdop = self.compute_GDOP(anchor_position, np.vstack([previous_measurements, new_measurements]))
        return gdop

    def evaluate_fim(self, new_measurements, anchor_position, previous_measurements):
        """
        Evaluate the FIM using the previous and new measurements.
        """
        # Calculate the FIM using the previous and new measurements
        
        new_measurements = np.array(new_measurements).reshape((-1, 3))

        fim_det_inverse = self.calculate_fim(anchor_position, np.vstack([previous_measurements, new_measurements]))

        return fim_det_inverse
    
    def compute_optimal_measurement_order(self, initial_position, optimal_waypoints, initial_trajectory):

        return optimal_waypoints
        

    def evaluate_path(self, anchor_estimate, past_measurement_points, new_measurement_points, remaining_trajectory):

        # calculate FIM using past and new measurement points and take the determinant

        # calculate the path length 

        pass
    
    def generate_initial_guess(self, previous_waypoints, last_measurement):
        """
        Generate an initial guess for the optimizer by randomly sampling within the bounds.
        """

        #return np.repeat(np.array(last_measurement), len(previous_waypoints) + 1)
        initial_guess = previous_waypoints.copy()

        n = len(previous_waypoints)



        if n == 0:
            return np.array(last_measurement)
        if n >= 1:
            initial_guess.append(np.array(last_measurement))
            return np.array(initial_guess)
        
        sum_x = sum(point[0] for point in previous_waypoints)
        sum_y = sum(point[1] for point in previous_waypoints)
        sum_z = sum(point[2] for point in previous_waypoints)

        centroid_x = sum_x / n
        centroid_y = sum_y / n
        centroid_z = sum_z / n

        initial_guess.append([centroid_x, centroid_y, centroid_z])
        return np.array(initial_guess)


    def optimize_waypoints_incrementally(self, anchor_estimate, previous_measurements, remaining_trajectory, bound_size = [5,5,5], max_waypoints=8, marginal_gain_threshold=0.01):
        best_waypoints = []
        previous_fim_det = self.calculate_fim(anchor_estimate, previous_measurements)
        previous_gdop = self.compute_GDOP(anchor_estimate, previous_measurements)

        last_measurement = previous_measurements[-1]
        previous_measurements = [[float('inf'), float('inf'), float('inf')]]
        bounds_base = [(coord-delta, coord+delta) for (coord,delta) in zip(last_measurement, bound_size)]
        

        
        
        for num_waypoints in range(1, max_waypoints + 1):

            bounds = bounds_base * num_waypoints

            # Generate an initial guess for the optimizer
            initial_guess = self.generate_initial_guess(best_waypoints, last_measurement).flatten()
            print(f"Initial guess: {initial_guess}")

            

            # Optimize for num_waypoints waypoints
            result = minimize(self.evaluate_gdop, initial_guess, args=(anchor_estimate, previous_measurements), method='L-BFGS-B', bounds=bounds)
    
            
            # Extract the new waypoints from the optimization result
            new_waypoints = result.x.reshape((-1, 3))
            # Calculate the determinant of the FIM with the new waypoints
            fim_det = self.calculate_fim(anchor_estimate, previous_measurements + new_waypoints.tolist())
            
            gdop = self.compute_GDOP(anchor_estimate, previous_measurements + new_waypoints.tolist())

            # Calculate the marginal gain
            marginal_gain = previous_fim_det - fim_det 
            marginal_gain_gdop = previous_gdop - gdop
            print(f"Marginal gain: {marginal_gain}, Marginal gain GDOP: {marginal_gain_gdop}")
            print(f"GDOP: {gdop}, previous GDOP: {previous_gdop}")
            print(f"FIM determinant: {fim_det}, previous FIM determinant: {previous_fim_det}")
            print(" ")

            
            if marginal_gain / previous_fim_det < marginal_gain_threshold:
                break
            
            if marginal_gain_gdop / previous_gdop < marginal_gain_threshold:
                pass

            # Update the best waypoints and previous FIM determinant
            best_waypoints = new_waypoints.tolist()
            previous_fim_det = fim_det
            previous_gdop = gdop

        return best_waypoints
    



def main():
    # Create an instance of the TrajectoryOptimization class
    trajectory_optimization = TrajectoryOptimization()

    def load_measurement_data_from_csv(path):

        def convert_str_to_list(s):
            # Replace 'nan', 'inf', and '-inf' with their corresponding numpy constants
            s = s.replace('nan', 'np.nan').replace('inf', 'np.inf').replace('-inf', '-np.inf')
            # Evaluate the string as a Python expression and return the result
            try:
                return eval(s)
            except Exception as e:
                # If evaluation fails, return the original string
                return s
    
        data = pd.read_csv(path, header=None, converters={i: convert_str_to_list for i in range(3)})
        data.columns = ['anchor_position_gt','measured_positions', 'measured_ranges']

        return data
    
    package_path = Path(__file__).parent.resolve()
    csv_dir = package_path / 'csv_files'

    df = load_measurement_data_from_csv(csv_dir / 'measurements.csv')

    previous_measurements = df.iloc[1]['measured_positions']
    anchor_gt = df.iloc[1]['anchor_position_gt']

    half_length = len(previous_measurements) // 5
    previous_measurements = previous_measurements[:half_length]

    indices = np.random.choice(len(previous_measurements), size=1, replace=False)
    indices.sort()
    
    # Sample measurements
    previous_measurements = np.array(previous_measurements)
    sampled_measurements = previous_measurements[indices]
    sampled_measurements = sampled_measurements.tolist()

    sampled_measurements = previous_measurements.tolist()
    waypoints = trajectory_optimization.optimize_waypoints_incrementally(anchor_estimate=anchor_gt, previous_measurements=sampled_measurements, remaining_trajectory=[], bound_size=[3,3,3], max_waypoints=8, marginal_gain_threshold=0.01)

    print(waypoints)
    
     # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Previous Measurements
    prev_meas = np.array(sampled_measurements)
    ax.scatter(prev_meas[:,0], prev_meas[:,1], prev_meas[:,2], color='blue', label='Previous Measurements')

    # Anchor Ground Truth
    ax.scatter(anchor_gt[0], anchor_gt[1], anchor_gt[2], color='red', s=100, label='Anchor Ground Truth')

    # Waypoints
    if waypoints:
        waypoints_arr = np.array(waypoints)
        ax.scatter(waypoints_arr[:,0], waypoints_arr[:,1], waypoints_arr[:,2], color='green', label='Additional Measurements')

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.show()



    # INPUT: 
    # array of positions used to estimate the position until now
    # estimated position of the anchor
    # remaining trajectory of the drone as a discretised array of positions (this is not the waypoints but directly the discretised path, can be used to calculate a deviation metric for example, or something else)
    
    
    # TO IMPLEMENT:
    # Objective function (given a path, output a metric) FIM ? GDOP ? COVARIANCE ?
    # Constraints - A cubic volume ?
    # Optimisation approach ? PSO, RRT, Genetic, SQP, 

    # Basically:
    # 1. Generate feasible paths
    # 2. Evaluate path wrt to the objective function (Optimality of the measurements but also path length)
    # 3. Select the best path

if __name__ == '__main__':
    main()