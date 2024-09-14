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
        self.method = "FIM"  # Method to use for optimization (FIM or GDOP)
        self.initial_guess = [0, 0, 0]  # Initial guess for the optimization algorithm
        self.bounds = [(-6,6),(-6,6),(0.2, 8)]# [(float('inf'), float('inf')), (float('inf'), float('inf')), (float('inf'), float('inf'))]  # Bounds for the optimization algorithm

    def calculate_fim(self, target_estimator, measurements, noise_variance):
        """
        Computes the Fisher Information Matrix (FIM) for the parameters (x0, y0, z0, gamma, beta)
        given the range measurements and noise characteristics.

        Parameters:
        x0, y0, z0: Coordinates of the object to estimate.
        gamma: Constant bias in the range measurements.
        beta: Linear bias in the range measurements.
        measurements: List of tuples [(x, y, z, r), ...] where (x, y, z) are the coordinates of known points
                    and r is the range measurement to the object.
        sigma_squared: The base variance of the noise in the measurements.

        Returns:
        FIM: Fisher Information Matrix (5x5 numpy array)
        """

        FIM = np.zeros((5, 5))

        var_x, var_y, var_z = noise_variance
        x0, y0, z0, gamma, beta = target_estimator

        # for measurement in measurements:
        #     xi, yi, zi = measurement
            
        #     # Calculate the true distance
        #     di = np.sqrt((xi - x0)**2 + (yi - y0)**2 + (zi - z0)**2)
            
        #     # Calculate the distance-dependent variance
        #     Cxi = var_x * (1 + di)**2
        #     Cyi = var_y * (1 + di)**2
        #     Czi = var_z * (1 + di)**2
            
        #     # Partial derivatives
        #     d_di_dx0 = (x0 - xi) / di
        #     d_di_dy0 = (y0 - yi) / di
        #     d_di_dz0 = (z0 - zi) / di   
            
        #     # Calculate the effective variance
        #     Ci = beta**2 * (d_di_dx0**2 / Cxi + d_di_dy0**2 / Cyi + d_di_dz0**2 / Czi)
            
        #     # Jacobian vector
        #     J = np.array([
        #         [-beta * d_di_dx0 / Cxi, -beta * d_di_dy0 / Cyi, -beta * d_di_dz0 / Czi, -1 / Ci, -di / Ci]
        #     ]).T
            
        #     # Fisher Information contribution from this measurement
        #     FIM_contrib = (1 / Ci) * np.dot(J, J.T)
            
        #     # Accumulate the Fisher Information Matrix
        #     FIM += FIM_contrib
        
        # # OPTION 2
        # FIM = np.zeros((3, 3))
    
        # for measurement in measurements:
        #     xi, yi, zi = measurement
            
        #     # Calculate the true distance
        #     di = np.sqrt((xi - x0)**2 + (yi - y0)**2 + (zi - z0)**2)
            
        #     # Partial derivatives for the distance with respect to x0, y0, z0
        #     d_di_dx0 = (x0 - xi) / di
        #     d_di_dy0 = (y0 - yi) / di
        #     d_di_dz0 = (z0 - zi) / di
            
        #     # Fisher Information contributions from this measurement
        #     FIM[0, 0] += (d_di_dx0**2) / var_x
        #     FIM[1, 1] += (d_di_dy0**2) / var_y
        #     FIM[2, 2] += (d_di_dz0**2) / var_z
        #     FIM[0, 1] += (d_di_dx0 * d_di_dy0) / np.sqrt(var_x * var_y)
        #     FIM[0, 2] += (d_di_dx0 * d_di_dz0) / np.sqrt(var_x * var_z)
        #     FIM[1, 2] += (d_di_dy0 * d_di_dz0) / np.sqrt(var_y * var_z)
        
        # # Since FIM is symmetric, we mirror the off-diagonal terms
        # FIM[1, 0] = FIM[0, 1]
        # FIM[2, 0] = FIM[0, 2]
        # FIM[2, 1] = FIM[1, 2]
        
        FIM = np.zeros((3, 3))  # Initialize the FIM matrix
        
        for measurement in measurements:
            x_i, y_i, z_i = measurement
            d_i = np.sqrt((x0 - x_i)**2 + (y0 - y_i)**2 + (z0 - z_i)**2)
            
            if d_i == 0:
                continue  # Avoid division by zero

            # Jacobian of the distance with respect to the anchor position
            jacobian = (np.array([x_i, y_i, z_i]) - np.array([x0, y0, z0])) / d_i

            # Update FIM
            FIM += (1 / (self.fim_noise_variance * (1 + d_i**2))) * np.outer(jacobian, jacobian)

        return FIM


    
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

    def evaluate_fim(self, new_measurements, anchor_position, anchor_variance, previous_measurements):
        """
        Evaluate the FIM using the previous and new measurements.
        """
        # Calculate the FIM using the previous and new measurements
        
        new_measurements = np.array(new_measurements).reshape((-1, 3))

        fim = self.calculate_fim(anchor_position, np.vstack([previous_measurements, new_measurements]), anchor_variance)
        fim_det_inverse = 1 / np.linalg.det(fim)
        return fim_det_inverse

    def cartesian_to_spherical(self, x, y, z, center):
        """Convert Cartesian coordinates to spherical coordinates"""
        x -= center[0]
        y -= center[1]
        z -= center[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi
    
    def spherical_to_cartesian(self, r, theta, phi, center):
        """Convert spherical coordinates to Cartesian coordinates"""
        x = r * np.sin(theta) * np.cos(phi) + center[0]
        y = r * np.sin(theta) * np.sin(phi) + center[1]
        z = r * np.cos(theta) + center[2]
        return x, y, z
        
    def get_spherical_bounds(self, center, radius):
        cartesian_bounds = self.bounds

        x_min, x_max = cartesian_bounds[0]
        y_min, y_max = cartesian_bounds[1]
        z_min, z_max = cartesian_bounds[2]
        x_c, y_c, z_c = center

        # Bounds for theta (from z-bounds)
        theta_min, theta_max = 0, np.pi  # Default bounds for theta

        if z_c > z_max:
            theta_min = None
            theta_max = None
        if z_c < z_min:
            theta_min = None
            theta_max = None

        if abs(z_c - z_max) < radius:
            theta_min = np.arccos((z_max - z_c) / radius)
        if abs(z_c - z_min) < radius:
            theta_max = np.arccos((z_min - z_c) / radius)

        # Bounds for phi (from x and y bounds)
        phi_min, phi_max = 0, 2 * np.pi  # Default bounds for phi

        if x_c > x_max:
            phi_min = None
            phi_max = None
        if x_c < x_min:
            phi_min = None
            phi_max = None
        
        if abs(x_c - x_max) < radius:
            phi_min = np.arccos((x_max - x_c) / radius)
        if abs(x_c - x_min) < radius:
            phi_max = np.arccos((x_min - x_c) / radius)

        if y_c > y_max:
            phi_min = None
            phi_max = None

        if y_c < y_min:
            phi_min = None
            phi_max = None
        
        if abs(y_c - y_max) < radius:
            phi_min = np.arccos((y_max - y_c) / radius)
        if abs(y_c - y_min) < radius:
            phi_max = np.arccos((y_min - y_c) / radius)


        return (theta_min, theta_max), (phi_min, phi_max)

    def optimize_waypoints_incrementally_spherical(self, anchor_estimator, anchor_estimate_variance, previous_measurements, remaining_trajectory, radius_of_search = 1, max_waypoints=8, marginal_gain_threshold=0.01):
        anchor_estimate = anchor_estimator[:3]
        
        anchor_estimate_variance = [var for var in anchor_estimate_variance]
        anchor_estimate_variance = [0.5 for var in anchor_estimate_variance]
        best_waypoints = []
        previous_fim_det = 1/np.linalg.det(self.calculate_fim(anchor_estimator, previous_measurements, anchor_estimate_variance))
        previous_gdop = self.compute_GDOP(anchor_estimate, previous_measurements)

        last_measurement = previous_measurements[-1]
        # Compute the directional vector between the last two measurements
        vector = np.array(previous_measurements[-1]) - np.array(previous_measurements[-2])
        # Use this vector to determine the initial guess for the new measurement, to be in the same direction as this
        initial_guess = self.cartesian_to_spherical(vector[0], vector[1], vector[2], [0,0,0])[1:]

        self.initial_guess = [previous_measurements[-1], self.spherical_to_cartesian(radius_of_search, initial_guess[0], initial_guess[1], previous_measurements[-1])]

        for i in range(max_waypoints):

            # On the first iteration, use double the radius of search
            current_radius = radius_of_search * 1 if i == 0 else radius_of_search

            def objective(new_measurement):
                new_measurement = self.spherical_to_cartesian(current_radius, new_measurement[0], new_measurement[1], last_measurement)
                new_measurement = np.array(new_measurement).reshape(1, -1)

                if self.method == "GDOP":
                    gdop = self.evaluate_gdop(new_measurement, anchor_estimate, previous_measurements)
                    return gdop
                elif self.method == "FIM":
                    fim_gain = self.evaluate_fim(new_measurement, anchor_estimator, anchor_estimate_variance, previous_measurements)
                    return fim_gain  # Objective: Minimize inverse FIM determinant

            # Define bounds for new measurement within the specified bounds
            bounds = [
                (0, np.pi),
                (0, 2*np.pi)
            ]

            bounds = self.get_spherical_bounds(last_measurement, current_radius)
            bounds_theta = bounds[0]
            bounds_psi = bounds[1]


            if None in bounds_theta or None in bounds_psi:
                    print("Optimisation not possible in the specified bounds")
                    break
            

            # Optimize for the next waypoint
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

            if result.success:
                new_waypoint = result.x
                new_waypoint = self.spherical_to_cartesian(current_radius ,new_waypoint[0], new_waypoint[1], last_measurement)

                if self.method == "GDOP":
                    new_gdop = self.evaluate_gdop(new_waypoint, anchor_estimate, previous_measurements)

                elif self.method == "FIM":
                    new_fim_det = self.evaluate_fim(new_waypoint, anchor_estimator, anchor_estimate_variance, previous_measurements)

                # Check marginal gain
                if self.method == "GDOP":
                    gdop_gain = previous_gdop - new_gdop

                    if gdop_gain/previous_gdop < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {gdop_gain/previous_gdop}. Stopping optimization.")
                        break
                elif self.method == "FIM":
                    fim_gain = previous_fim_det - new_fim_det

                    if fim_gain/previous_fim_det < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {fim_gain/previous_fim_det}. Stopping optimization.")
                        break

                # Update best waypoints and metrics
                best_waypoints.append(new_waypoint)
                previous_measurements = np.vstack([previous_measurements, new_waypoint])
                if self.method == "GDOP":
                    previous_gdop = new_gdop
                elif self.method == "FIM":
                    previous_fim_det = new_fim_det
                last_measurement = new_waypoint

            else:
                print(f"Optimization failed at waypoint {i + 1}")
                break

            # Initial guess for the next waypoint
            initial_guess = result.x
            self.initial_guess.append(self.spherical_to_cartesian(radius_of_search, initial_guess[0], initial_guess[1], last_measurement))

        return best_waypoints
                

    def optimize_waypoints_incrementally_cubical(self, anchor_estimator, anchor_estimate_variance, previous_measurements, remaining_trajectory, bound_size=[5,5,5], max_waypoints=8, marginal_gain_threshold=0.01):
        anchor_estimate = anchor_estimator[:3]
        
        best_waypoints = []
        previous_fim_det = self.calculate_fim(anchor_estimate, anchor_estimate_variance, previous_measurements)
        previous_gdop = self.compute_GDOP(anchor_estimate, previous_measurements)

        last_measurement = previous_measurements[-1]

        for i in range(max_waypoints):
            def objective(new_measurement):
                new_measurement = np.array(new_measurement).reshape(1, -1)
                if self.method == "GDOP":
                    gdop = self.evaluate_gdop(new_measurement, anchor_estimate, previous_measurements)
                    return gdop
                elif self.method == "FIM":
                    fim_gain = self.evaluate_fim(new_measurement, anchor_estimator, anchor_estimate_variance, previous_measurements)
                    return fim_gain  # Objective: Minimize inverse FIM determinant

            # Define bounds for new measurement within the specified bounds
            bounds = [
                (last_measurement[0] - bound_size[0]/2, last_measurement[0] + bound_size[0]/2),
                (last_measurement[1] - bound_size[1]/2, last_measurement[1] + bound_size[1]/2),
                (last_measurement[2] - bound_size[2]/2, last_measurement[2] + bound_size[2]/2)
            ]

            # Initial guess for the next waypoint
            initial_guess = last_measurement + np.random.uniform(-bound_size[0]/4, bound_size[0]/4, size=3)

            # Optimize for the next waypoint
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

            if result.success:
                new_waypoint = result.x
                if self.method == "GDOP":
                    new_gdop = self.evaluate_gdop(new_waypoint, anchor_estimate, previous_measurements)

                elif self.method == "FIM":
                    new_fim_det = self.evaluate_fim(new_waypoint, anchor_estimator, anchor_estimate_variance, previous_measurements)

                # Check marginal gain
                if self.method == "GDOP":
                    gdop_gain = previous_gdop - new_gdop
                    if gdop_gain/previous_gdop < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {gdop_gain/previous_gdop}. Stopping optimization.")
                        break
                elif self.method == "FIM":
                    fim_gain = previous_fim_det - new_fim_det

                    if fim_gain/previous_fim_det < marginal_gain_threshold:
                        print(f"Marginal gain below threshold: {fim_gain/previous_fim_det}. Stopping optimization.")
                        break

                # Update best waypoints and metrics
                best_waypoints.append(new_waypoint)
                previous_measurements = np.vstack([previous_measurements, new_waypoint])
                if self.method == "GDOP":
                    previous_gdop = new_gdop
                elif self.method == "FIM":
                    previous_fim_det = new_fim_det
                last_measurement = new_waypoint

            else:
                print(f"Optimization failed at waypoint {i + 1}")
                break

        return best_waypoints
    
    def compute_new_mission_waypoints(self, initial_position, initial_remaining_waypoints, optimal_waypoints, method="return_to_closest"):
        """Compute the new mission waypoints based on the initial position, initial waypoints, and optimal waypoints
        
        Parameters:
        - initial_position: list of floats, the initial position of the drone [x, y, z]
        - initial_waypoints: list of lists, the initial waypoints to follow
        - optimal_waypoints: list of lists, the optimal waypoints to follow
        - method: str, the method to use to compute the new mission waypoints
        
        Returns:
        - new_mission_waypoints: list of lists, the new mission waypoints to follow
        """
        
        new_mission_waypoints = []
        
        def closest_point_on_line(A, B, C):
            # Convert points to numpy arrays
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)
            
            # Vector AB and AC
            AB = B - A
            AC = C - A
            
            # Project AC onto AB
            t = np.dot(AC, AB) / np.dot(AB, AB)
            
            # Find the closest point P on the line
            P = A + t * AB
            
            return P

        if method == "return_to_initial":
            new_mission_waypoints.extend(optimal_waypoints)
            new_mission_waypoints.append(initial_position)
            new_mission_waypoints.extend(initial_remaining_waypoints)
            link_waypoints = [initial_position]

        if method == "straight_to_wapoint":
            new_mission_waypoints.extend(optimal_waypoints)
            new_mission_waypoints.extend(initial_remaining_waypoints)
            link_waypoints = []

        if method == "return_to_closest":
            closest_point = closest_point_on_line(initial_position, initial_remaining_waypoints[0], optimal_waypoints[-1])
            new_mission_waypoints.extend(optimal_waypoints)
            new_mission_waypoints.append(closest_point)
            new_mission_waypoints.extend(initial_remaining_waypoints)
            link_waypoints = [closest_point]
        
        return new_mission_waypoints, optimal_waypoints, link_waypoints, initial_remaining_waypoints




        optimal_trajectory = self.create_optimal_trajectory(initial_position, optimal_points)
        link_trajectory = self.create_link_from_deviation_to_initial_trajectory(initial_trajectory, optimal_points)

        trajectory = Trajectory()
        trajectory.spline_x = np.concatenate((optimal_trajectory.spline_x, link_trajectory.spline_x))
        trajectory.spline_y = np.concatenate((optimal_trajectory.spline_y, link_trajectory.spline_y))
        trajectory.spline_z = np.concatenate((optimal_trajectory.spline_z, link_trajectory.spline_z))

        return trajectory


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
    
    csv_index = 0
    previous_measurements = df.iloc[csv_index]['measured_positions']
    anchor_gt = df.iloc[csv_index]['anchor_position_gt']

    half_length = len(previous_measurements) * 5 // 6
    previous_measurements = previous_measurements[:half_length]

    indices = np.random.choice(len(previous_measurements), size=1, replace=False)
    indices.sort()
    
    # Sample measurements
    previous_measurements = np.array(previous_measurements)
    sampled_measurements = previous_measurements[indices]
    sampled_measurements = sampled_measurements.tolist()

    sampled_measurements = previous_measurements.tolist()
    #waypoints = trajectory_optimization.optimize_waypoints_incrementally_cubical(anchor_estimate=anchor_gt, previous_measurements=sampled_measurements, remaining_trajectory=[], bound_size=[3,3,3], max_waypoints=20, marginal_gain_threshold=0.01)
    waypoints = trajectory_optimization.optimize_waypoints_incrementally_spherical(anchor_estimate=anchor_gt, previous_measurements=sampled_measurements, remaining_trajectory=[], radius_of_search=2, max_waypoints=20, marginal_gain_threshold=0.05)

    
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
    guess = trajectory_optimization.initial_guess
    guess = np.array(guess)

    ax.scatter(guess[1:, 0], guess[1:, 1], guess[1:, 2], color='black', label='Initial Guess')

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