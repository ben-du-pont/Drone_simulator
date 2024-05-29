import numpy as np
from scipy.optimize import least_squares

from drone_uwb_simulator.UWB_protocol import Anchor
from online_uwb_initialisation.trajectory_optimisation import NonLinearTrajectoryOptimization

from drone_uwb_simulator.drone_simulator import DroneSimulation

from drone_uwb_simulator.drone_dynamics import Waypoint, Trajectory


class UwbOnlineInitialisation:
    def __init__(self):

        self.base_anchors = []
        self.unknown_anchors = []

        self.drone_postion = []

        self.known_anchor_measurements = {}
        self.unknown_anchor_measurements = {}

        self.optimiser = NonLinearTrajectoryOptimization()

        self.anchor_rough_estimate_found = False
        self.trajectory_optimiser_done = False
        self.all_optimal_measurements_collected = False

        self.trajectory_waypoints = []
        self.optimised_trajectory_waypoints = []

        self.spline_x = None
        self.spline_y = None
        self.spline_z = None

        self.gdop_threshold = 1.0
        self.distance_to_anchor_ratio_threshold = 0.05
        self.number_of_measurements = 2
        self.distance_rejection_threshold = 50

        self.error = 100

    def process_anchor_info(self, known_anchors, unknown_anchors):
        """Process the anchor information and store it in the class variables"""
        self.base_anchors = []
        self.unknown_anchors = []

        for anchor in known_anchors:
            self.base_anchors.append(anchor)
        for anchor in unknown_anchors:
            self.unknown_anchors.append(anchor)

    def get_distance_to_anchors(self, x, y, z):
        """Given a drone position, returns the distance to all the anchors it can reach and their IDs"""
        anchors_distances = []
        unknown_anchor_distances = []

        for anchor in self.base_anchors:
            distance = anchor.request_distance(x, y, z)
            anchors_distances.append([distance, anchor.anchor_ID])

        for unknown_anchor in self.unknown_anchors:
            distance = unknown_anchor.request_distance(x, y, z)
            unknown_anchor_distances.append([distance, anchor.anchor_ID])

        return anchors_distances, unknown_anchor_distances
    
    def compute_GDOP(self, measurements):
        """Compute the Geometric Dilution of Precision (GDOP) given a set of measurements corresponding to drone positions in space"""

        first_key = next(iter(self.unknown_anchor_measurements))
        measurement_points = self.unknown_anchor_measurements[first_key]["positions_pre_rough_estimate"] + self.unknown_anchor_measurements[first_key]["positions_post_rough_estimate"]
        if len(measurement_points) < 2:
            return float('inf') # Not enough points to calculate GDOP
        A = np.hstack((measurement_points, np.ones((np.array(measurement_points).shape[0], 1))))  # Augmenting with ones for affine transformations
        try:
            inv_at_a = np.linalg.inv(A.T @ A)
            gdop = np.sqrt(np.trace(inv_at_a))
            if gdop is not None:
                return gdop
            else:
                return float('inf')
        except np.linalg.LinAlgError:
            return float('inf')  # Matrix is singular, cannot compute GDOP

    def estimate_anchor_position_non_linear_least_squares(self, measurements, initial_guess=[0, 0, 0, 0, 0]):
        """Estimate the position of an anchor given distance measurements fron different drone positions using linear least squares optimization"""

        # Define the loss function and its Jacobian
        def loss_function(x0, measurements):
            errors = []
            J = []  # Jacobian matrix
            guess = x0[0:3]
            bias = x0[3]
            linear_bias = x0[4]
            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                distance_vector = np.array([guess[0] - x, guess[1] - y, guess[2] - z])
                distance = np.linalg.norm(distance_vector)
                estimated_dist = distance * (1 + linear_bias) + bias
                error = estimated_dist - measured_dist
                errors.append(error)
                
                # Compute partial derivatives for the Jacobian
                if distance != 0:
                    partial_x = (guess[0] - x) / distance
                    partial_y = (guess[1] - y) / distance
                    partial_z = (guess[2] - z) / distance
                else:
                    partial_x = partial_y = partial_z = 0
                
                J.append([partial_x * (1 + linear_bias), partial_y * (1 + linear_bias), partial_z * (1 + linear_bias), 1, distance])

            return errors, np.array(J)

        # Perform least squares optimization and calculate Jacobian at solution
        result = least_squares(lambda x0, measurements: loss_function(x0, measurements)[0], initial_guess, args=(measurements,), jac=lambda x0, measurements: loss_function(x0, measurements)[1], method='lm')

        # Calculate covariance matrix from the Jacobian at the last evaluated point
        J = result.jac
        cov_matrix = np.linalg.pinv(J.T @ J)  # Moore-Penrose pseudo-inverse of J.T @ J

        return result.x, cov_matrix
    
    def estimate_anchor_position_linear_least_squares(self, measurements, initial_guess=[0, 0, 0, 0, 0]):

        """Estimate the position of an anchor given distance measurements fron different drone positions using linear least squares optimization"""
        A = []
        b = []
        for measurement in measurements:
            x, y, z, measured_dist = measurement[0:4]

            A.append([-2*x, -2*y, -2*z, np.linalg.norm([x, y, z])**2, 2*measured_dist**2, 1])
            b.append(measured_dist**2)

        x = np.linalg.lstsq(A, b, rcond=None)[0]

        linear_bias = x[3]
        bias = x[4]
        position = [pos/linear_bias for pos in x[:3]]

        return position + [bias, linear_bias], None



        A = measurements
    
    
    def measurement_callback(self, drone_position, drone_progress_waypoint_idx):
        """Callback function that is called every time a new drone position is reached, which triggers the UWB measurements"""
        drone_x, drone_y, drone_z = drone_position

        # Get new distance measurements from the anchors
        anchors_distances, unknown_anchor_distances = self.get_distance_to_anchors(drone_x, drone_y, drone_z)


        # 1st step, rough initialisation of the unknown anchors
        if not self.anchor_rough_estimate_found:
            for distance, anchor_id in unknown_anchor_distances: # For now, this will run once since we have only one unknown anchor, but later, will probably need to change the order of the loops or something

                anchor_distances = self.keep_or_discard_measurement(drone_position, distance, anchor_id)

                # We added a new measuremnet, we can now compute the GDOP and update it in the dictionary for this specific anchor
                measurements = None
                gdop = self.compute_GDOP(measurements)
                anchor_distances["gdop_rough"] = gdop
                
                # If the GDOP is below the threshold, we can run the linear and non linear least sqaures optimisation to get a rough estimate of the anchor position
                if gdop < self.gdop_threshold:
                    self.anchor_rough_estimate_found = True

                    measurements = []
                    for distance, position in zip(anchor_distances["distances_pre_rough_estimate"]+anchor_distances["distances_post_rough_estimate"], anchor_distances["positions_pre_rough_estimate"] + anchor_distances["positions_post_rough_estimate"]):
                        x, y, z = position
                        measurements.append([x, y, z, distance])

                    estimator, covariance_matrix = self.estimate_anchor_position_linear_least_squares(measurements)

                    anchor_distances["estimator_rough_linear"] = estimator
                    anchor_distances["estimated_position_rough_linear"] = np.array(anchor_distances["estimator_rough_linear"][:3])

                    # estimator, covariance_matrix = self.estimate_anchor_position_non_linear_least_squares(measurements, anchor_distances["estimator_rough_linear"])

                    anchor_distances["estimator_rough_non_linear"] = estimator
                    anchor_distances["estimated_position_rough_non_linear"] = np.array(anchor_distances["estimator_rough_non_linear"][:3])

                    self.error = self.calculate_position_error(self.unknown_anchors[0].get_anchor_coordinates() ,anchor_distances["estimated_position_rough_non_linear"])
                    # 2nd step: optimise the trajectory
                    remaining_waypoints = self.trajectory_waypoints[drone_progress_waypoint_idx:]

                    self.optimiser.update_anchor_positions(anchor_distances["estimated_position_rough_non_linear"])

                    self.optimiser.update_waypoints(remaining_waypoints)

                    # self.optimiser.update_weights(weights)



                    self.optimised_trajectory_waypoints = self.optimiser.run_optimization()

                    self.spline_x, self.spline_y, self.spline_z = self.optimiser.create_optimised_trajectory(self.drone_postion ,self.optimised_trajectory_waypoints, drone_progress_waypoint_idx, self.trajectory_waypoints)
    
        elif not self.all_optimal_measurements_collected:
            
            if drone_position in self.optimised_trajectory_waypoints: # The drone is at one of the optimal waypoints
                for distance, anchor_id in unknown_anchor_distances:
                    anchor_distances = self.unknown_anchor_measurements.setdefault(anchor_id, {"distances_pre_rough_estimate": [], "positions_pre_rough_estimate": [], "distances_post_rough_estimate": [], "positions_post_rough_estimate": [], "estimator_rough_linear": [0, 0, 0, 0, 0], "estimated_position_rough_linear": [], "estimator_rough_non_linear": [0, 0, 0, 0, 0], "estimated_position_rough_non_linear": [], "estimator": [0, 0, 0, 0, 0], "estimated_position": [], "gdop_rough": float('inf'),"gdop_full": float('inf')})

                    anchor = self.unknown_anchors[0]
                    for i in range(self.number_of_measurements):
                        
                        distance = anchor.request_distance(drone_x, drone_y, drone_z)

                        anchor_distances["distances_post_rough_estimate"].append(distance)
                        anchor_distances["positions_post_rough_estimate"].append((drone_x, drone_y, drone_z))

                if (np.array(drone_position) == self.optimised_trajectory_waypoints[-2]).all():
                    self.all_optimal_measurements_collected = True
                    

                    measurements = []
                    for distance, position in zip(anchor_distances["distances_pre_rough_estimate"]+anchor_distances["distances_post_rough_estimate"], anchor_distances["positions_pre_rough_estimate"] + anchor_distances["positions_post_rough_estimate"]):
                        x, y, z = position
                        measurements.append([x, y, z, distance])

                    estimator, covariance_matrix = self.estimate_anchor_position_linear_least_squares(measurements)

                    anchor_distances["estimator"] = estimator
                    anchor_distances["estimated_position"] = np.array(anchor_distances["estimator"][:3])

                    self.error = self.calculate_position_error(self.unknown_anchors[0].get_anchor_coordinates(), anchor_distances["estimated_position"])

        else:
            pass


    def keep_or_discard_measurement(self, drone_position, distance, anchor_id):
        drone_x, drone_y, drone_z = drone_position

        anchor_distances = self.unknown_anchor_measurements.setdefault(anchor_id, {"distances_pre_rough_estimate": [], "positions_pre_rough_estimate": [], "distances_post_rough_estimate": [], "positions_post_rough_estimate": [], "estimator_rough_linear": [0, 0, 0, 0, 0], "estimated_position_rough_linear": [], "estimator_rough_non_linear": [0, 0, 0, 0, 0], "estimated_position_rough_non_linear": [], "estimator": [0, 0, 0, 0, 0], "estimated_position": [], "gdop_rough": float('inf'),"gdop_full": float('inf')})

        # Reject measurements that are too far away
        if distance > self.distance_rejection_threshold:
            return anchor_distances
        
        
        # I we have no measurements yet, or if we decide to gather mutiple measurements at points very close to each other for robustness, add the measurement to the dictionnary
        if (len(anchor_distances["distances_pre_rough_estimate"]) % self.number_of_measurements != 0) or len(anchor_distances["distances_pre_rough_estimate"])  == 0:

            anchor_distances["distances_pre_rough_estimate"].append(distance)
            anchor_distances["positions_pre_rough_estimate"].append((drone_x, drone_y, drone_z))

        # If the drone has moved significantly since the last measurement, add the new measurement to the dictionnary
        elif np.linalg.norm(np.array(anchor_distances["positions_pre_rough_estimate"][-1]) - np.array((drone_x, drone_y, drone_z))) / distance > self.distance_to_anchor_ratio_threshold:
            anchor_distances["distances_pre_rough_estimate"].append(distance)
            anchor_distances["positions_pre_rough_estimate"].append((drone_x, drone_y, drone_z))

        return anchor_distances

    

    

    
    






    def calculate_position_error(self, real_positions, estimated_positions):
            """Calculate the error between the estimated anchor position and the real ground truth anchor position
            Can be used for one or multiple anchors"""

            actual_position = np.array([(a.x, a.y, a.z) for a in self.unknown_anchors])
            error = np.linalg.norm(estimated_positions - actual_position)
            return error
    
    def calculate_estimator_error(self, real_anchor_values, estimator):
        """Calculate the error between the estimated anchor estimator and the real ground truth anchor characteristics
            Can be used for one or multiple anchors
            The estimator contains the position of the anchor and the bias and linear bias"""
        
        actual_position = np.array([(a.x, a.y, a.z, a.bias, a.linear_bias) for a in self.unknown_anchors])
        error = np.linalg.norm(estimator - actual_position)

        return error
    








    ### Helper functions for quickly computing the anchor initialisation without relying on the simulator to run real time

    
    def reset_measurements_pre_rough_initialisation(self, anchor_id):
        """Reset the measurements in the dictionnary that were gathered for the rough estimation of the anchor"""

        self.unknown_anchor_measurements[anchor_id]["distances_pre_rough_estimate"] = []
        self.unknown_anchor_measurements[anchor_id]["positions_pre_rough_estimate"] = []
        self.unknown_anchor_measurements[anchor_id]["estimator_rough"] = [0, 0, 0, 0, 0]
        self.unknown_anchor_measurements[anchor_id]["estimated_position_rough"] = []


    def reset_measurements_post_rough_initialisation(self, anchor_id):
        """Reset the measurements in the dictionnary that were gathered after the rough estimation of the anchor from the optimal waypoints"""

        self.unknown_anchor_measurements[anchor_id]["distances_post_rough_estimate"] = []
        self.unknown_anchor_measurements[anchor_id]["positions_post_rough_estimate"] = []
        self.unknown_anchor_measurements[anchor_id]["estimator"] = [0, 0, 0, 0, 0]
        self.unknown_anchor_measurements[anchor_id]["estimated_position"] = []
        
    def reset_all_measurements(self, anchor_id):
        """Reset all the measurements in the dictionnary for a specific anchor"""

        self.unknown_anchor_measurements[anchor_id] = {"distances_pre_rough_estimate": [], "positions_pre_rough_estimate": [], "distances_post_rough_estimate": [], "positions_post_rough_estimate": [], "estimator_rough_linear": [0, 0, 0, 0, 0], "estimated_position_rough_linear": [], "estimator_rough_non_linear": [0, 0, 0, 0, 0], "estimated_position_rough_non_linear": [], "estimator": [0, 0, 0, 0, 0], "estimated_position": [], "gdop_rough": float('inf'),"gdop_full": float('inf')}
    



    def compute_anchor_rough_estimate(self, drone_trajectory: Trajectory, anchor: Anchor):

        anchor_distances = self.unknown_anchor_measurements.setdefault(anchor.anchor_ID, {"distances_pre_rough_estimate": [], "positions_pre_rough_estimate": [], "distances_post_rough_estimate": [], "positions_post_rough_estimate": [], "estimator_rough_linear": [0, 0, 0, 0, 0], "estimated_position_rough_linear": [], "estimator_rough_non_linear": [0, 0, 0, 0, 0], "estimated_position_rough_non_linear": [], "estimator": [0, 0, 0, 0, 0], "estimated_position": [], "gdop_rough": float('inf'),"gdop_full": float('inf')})

        self.reset_all_measurements(anchor.anchor_ID)

        for x,y,z in zip(drone_trajectory.spline_x, drone_trajectory.spline_y, drone_trajectory.spline_z):
            distance = anchor.request_distance(x, y, z)
            drone_position = [x, y, z]
            anchor_distances = self.keep_or_discard_measurement(drone_position, distance, anchor.anchor_ID)

            measurements = None
            gdop = self.compute_GDOP(measurements)
            anchor_distances["gdop_rough"] = gdop
            
            # If the GDOP is below the threshold, we can run the linear and non linear least sqaures optimisation to get a rough estimate of the anchor position
            if gdop < self.gdop_threshold:
                self.anchor_rough_estimate_found = True

                measurements = []
                for distance, position in zip(anchor_distances["distances_pre_rough_estimate"]+anchor_distances["distances_post_rough_estimate"], anchor_distances["positions_pre_rough_estimate"] + anchor_distances["positions_post_rough_estimate"]):
                    x, y, z = position
                    measurements.append([x, y, z, distance])

                estimator, covariance_matrix = self.estimate_anchor_position_linear_least_squares(measurements)

                anchor_distances["estimator_rough_linear"] = estimator
                anchor_distances["estimated_position_rough_linear"] = np.array(anchor_distances["estimator_rough_linear"][:3])

                estimator, covariance_matrix = self.estimate_anchor_position_non_linear_least_squares(measurements, anchor_distances["estimator_rough_linear"])

                anchor_distances["estimator_rough_non_linear"] = estimator
                anchor_distances["estimated_position_rough_non_linear"] = np.array(anchor_distances["estimator_rough_non_linear"][:3])

                self.error = self.calculate_position_error(self.unknown_anchors[0].get_anchor_coordinates() ,anchor_distances["estimated_position_rough_non_linear"])

                return drone_position
            
        return None



    def populate_measurements_from_waypoints(self, waypoints, unknown_anchor):
        anchor_distances = self.unknown_anchor_measurements.setdefault(unknown_anchor.anchor_ID, {"distances_pre_rough_estimate": [], "positions_pre_rough_estimate": [], "distances_post_rough_estimate": [], "positions_post_rough_estimate": [], "estimator_rough_linear": [0, 0, 0, 0, 0], "estimated_position_rough_linear": [], "estimator_rough_non_linear": [0, 0, 0, 0, 0], "estimated_position_rough_non_linear": [], "estimator": [0, 0, 0, 0, 0], "estimated_position": [], "gdop_rough": float('inf'),"gdop_full": float('inf')})
        for waypoint in waypoints:
            x,y,z = waypoint

            for i in range(self.number_of_measurements):
                distance_to_anchor = unknown_anchor.request_distance(x, y, z)
                anchor_distances["distances_post_rough_estimate"].append(distance_to_anchor)
                anchor_distances["positions_post_rough_estimate"].append((x, y, z))


    
    def refine_anchor_positions(self, waypoints, unknown_anchor):

        self.populate_measurements_from_waypoints(waypoints, unknown_anchor)
        anchor_distances = self.unknown_anchor_measurements[next(iter(self.unknown_anchor_measurements))]
        measurements = []
        for distance, position in zip(anchor_distances["distances_pre_rough_estimate"]+anchor_distances["distances_post_rough_estimate"], anchor_distances["positions_pre_rough_estimate"] + anchor_distances["positions_post_rough_estimate"]):
            x, y, z = position
            measurements.append([x, y, z, distance])

        estimator, covariance_matrix = self.estimate_anchor_position_non_linear_least_squares(measurements,initial_guess=anchor_distances["estimator_rough_non_linear"])

        anchor_distances["estimator"] = estimator
        anchor_distances["estimated_position"] = np.array(anchor_distances["estimator"][:3])
