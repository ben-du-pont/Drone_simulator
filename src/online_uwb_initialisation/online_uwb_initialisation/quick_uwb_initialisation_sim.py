from drone_uwb_simulator.UWB_protocol import Anchor
from drone_uwb_simulator.drone_simulator import DroneSimulation
from drone_uwb_simulator.drone_dynamics import Waypoint, Trajectory

from online_uwb_initialisation.uwb_online_initialisation import UwbOnlineInitialisation
from online_uwb_initialisation.trajectory_optimisation import TrajectoryOptimization


import pandas as pd
import numpy as np
import random
import csv
import copy

import matplotlib.pyplot as plt
from matplotlib import gridspec

from pathlib import Path

from sklearn.mixture import GaussianMixture

package_path = Path(__file__).parent.resolve()
csv_dir = package_path / 'csv_files'

# For formatting the numpy arrays as strings in the csv file
def array_to_string(array):
    return str(array).replace('\n', '').replace('[', '').replace(']', '')




class CalculateOnlineInitialisation:

    def __init__(self):
        """Initialise the CalculateOnlineInitialisation object."""

        # Create the DroneSimulation object, the UwbOnlineInitialisation object, and the TrajectoryOptimization object
        self.drone_sim = DroneSimulation()
        self.uwb_online_initialisation = UwbOnlineInitialisation()
        self.optimiser = TrajectoryOptimization()

        # Initialise the measurement vector and the ground truth measurement vector
        self.measurement_vector = {}
        self.measurement_vector_gt = {}
        self.measurement_vector_optimal = {}

        # Initialise the vectors for the stopping criterion metrics
        self.gdop_vector = {}
        self.fim_vector = {}
        self.condition_number_vector = {}
        self.covariances_vector = {}
        self.residuals_vector = {}
        self.verifications_vector = {}
        self.pos_delta_vector = {}

        self.weight_vector = {}

        self.number_of_measurements = {}

        # Initialise the error vector
        self.constant_bias_vector = {}
        self.linear_bias_vector = {}
        self.error_vector = {}
        self.constant_bias_error_vector = {}
        self.linear_bias_error_vector = {}

        self.full_trajectory_x, self.full_trajectory_y, self.full_trajectory_z = [], [], []
        self.waypoints = []

        self.optimal_trajectories = {}
        self.optimal_waypoints = {}

        self.drone_trajectory = None

        self.randomise_environment()

    def randomise_environment(self):
        """Function to randomise the environment by adjusting the anchor parameters and the measurement collection procedure."""

        # Randomise the anchor
        for index, anchor in enumerate(self.drone_sim.unknown_anchors):
            self.drone_sim.unknown_anchors[index].bias = random.uniform(0, 0.3)
            self.drone_sim.unknown_anchors[index].linear_bias = random.uniform(1, 1.03)
            self.drone_sim.unknown_anchors[index].noise_variance = random.uniform(0, 0.2)
            self.drone_sim.unknown_anchors[index].outlier_probability = random.uniform(0, 0.2)

        # Randomise the measurement collection delta
        self.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold'] = 0.03 #random.uniform(0.01, 0.05)

        waypoints = [waypoint.get_coordinates() for waypoint in self.drone_sim.waypoints]
        
        self.uwb_online_initialisation.passed_waypoints = waypoints[0]
        self.uwb_online_initialisation.remaining_waypoints = waypoints[1:]




    # Env setup for the drone halle experiment
    def setup_environment_drone_halle(self):
        """Function to randomise the environment by adjusting the anchor parameters and the measurement collection procedure."""
        anchor_positions = [[-2.502, -1.725, 0.223], 
                            [3.052, -1.378, 0.888], 
                            [-2.764, 1.080, 1.988] , 
                            [0.870, 3.464, 2.129]] # [2.870, 2.464, 2.129]
        self.drone_sim.unknown_anchors = []

        for anchor_id in range(1,5):
            # Randomise the anchor
            anchor = Anchor(str(anchor_id), *anchor_positions[anchor_id-1])
            anchor.bias = random.uniform(-0.05, 0.05)
            anchor.linear_bias = random.uniform(1, 1.02)
            anchor.noise_variance = random.uniform(0.0, 0.1)
            anchor.outlier_probability = random.uniform(0.0, 0.1)
            self.drone_sim.unknown_anchors.append(anchor)


        # Randomise the measurement collection delta
        self.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold'] = 0.05 #random.uniform(0.01, 0.05)

    def setup_trajectory_drone_halle(self):
        self.drone_sim.drone_trajectory = Trajectory(1, 0.05)

        trajectory_height = 1
        start_position = [1.25, -2.8, 0]
        hover_position = [1.25, -2.8, trajectory_height]
        intermediate_position = [2, -1.7, trajectory_height]
        opposite_position = [-1.25, 2.8, trajectory_height]
        hover_position2 = [1.25, -2.8, trajectory_height]
        end_position = [1.25, -2.8, 0]

        self.waypoints = [start_position, hover_position, intermediate_position, opposite_position, hover_position2, end_position]

        ordered_waypoints = [Waypoint(*start_position), Waypoint(*hover_position), Waypoint(*intermediate_position), Waypoint(*opposite_position),  Waypoint(*hover_position2), Waypoint(*end_position)]

        self.drone_sim.drone_trajectory.construct_trajectory_linear(ordered_waypoints)


    # Always call this function on all anchors before running the pipeline
    def reset_metrics(self, anchor_id):
        """Reset the metrics vectors and the measurement vector.
        
        Parameters:
        anchor_id (int): The ID of the anchor for which the metrics are computed amd should be reset.
        """
        self.measurement_vector.setdefault(anchor_id, [])
        self.measurement_vector_gt.setdefault(anchor_id, [])
        self.measurement_vector_optimal.setdefault(anchor_id, [])

        self.gdop_vector.setdefault(anchor_id, [])
        self.fim_vector.setdefault(anchor_id, [])
        self.condition_number_vector.setdefault(anchor_id, [])
        self.covariances_vector.setdefault(anchor_id, [])
        self.residuals_vector.setdefault(anchor_id, [])
        self.verifications_vector.setdefault(anchor_id, [])
        self.pos_delta_vector.setdefault(anchor_id, [])

        self.error_vector.setdefault(anchor_id, [])
        self.constant_bias_error_vector.setdefault(anchor_id, [])
        self.linear_bias_error_vector.setdefault(anchor_id, [])
        self.constant_bias_vector.setdefault(anchor_id, [])
        self.linear_bias_vector.setdefault(anchor_id, [])

        self.number_of_measurements.setdefault(anchor_id, 0)

        self.weight_vector.setdefault(anchor_id, [])

        self.uwb_online_initialisation.reset_all_measurements(anchor_id)


    # Functions to gather measurements that are constant (for comparison between methods with the same data, and not random small differences)
    def gather_measurements(self):

        self.full_trajectory_x, self.full_trajectory_y, self.full_trajectory_z = self.drone_sim.drone_trajectory.spline_x, self.drone_sim.drone_trajectory.spline_y, self.drone_sim.drone_trajectory.spline_z

        for p_x, p_y, p_z in zip(self.drone_sim.drone_trajectory.spline_x, self.drone_sim.drone_trajectory.spline_y, self.drone_sim.drone_trajectory.spline_z):
            
            for anchor in self.drone_sim.unknown_anchors:
                anchor_ID = anchor.anchor_ID
                distance = anchor.request_distance(p_x, p_y, p_z)
                self.measurement_vector[anchor_ID].append([p_x, p_y, p_z, distance])

    def gather_measurements_ground_truth(self):

        self.full_trajectory_x, self.full_trajectory_y, self.full_trajectory_z = self.drone_sim.drone_trajectory.spline_x, self.drone_sim.drone_trajectory.spline_y, self.drone_sim.drone_trajectory.spline_z

        for p_x, p_y, p_z in zip(self.drone_sim.drone_trajectory.spline_x, self.drone_sim.drone_trajectory.spline_y, self.drone_sim.drone_trajectory.spline_z):
            
            for anchor in self.drone_sim.unknown_anchors:
                anchor_ID = anchor.anchor_ID
                distance = anchor.request_distance(p_x, p_y, p_z)
                self.measurement_vector_gt[anchor_ID].append([p_x, p_y, p_z, distance])

    def gather_measurements_on_optimal_trajectory(self, trajectory, anchor_id):
    
        """Function to gather measurements from the drone trajectory and the unknown anchors.
        The measurements are stored in the unknown_anchor_measurements dictionary in the UwbOnlineInitialisation object.
        They are also preprocessed, i.e only added if they are within the distance_to_anchor_ratio_threshold and close enough to the drone.
        It also adds the measurements to the measurement vector_optimal for the optimal trajectory."""

        anchor = next(anchor for anchor in self.drone_sim.unknown_anchors if anchor.anchor_ID == anchor_id)

        for p_x, p_y, p_z in zip(trajectory.spline_x, trajectory.spline_y, trajectory.spline_z):
            
            distance = anchor.request_distance(p_x, p_y, p_z)
            self.measurement_vector_optimal[anchor_id].append([p_x, p_y, p_z, distance])


    # Easy way to write settings to the UWB online initialisation object depending on the methods to test
    def write_settings(self, setting):

        if setting.startswith("filtered"):
            self.uwb_online_initialisation.params['outlier_removing'] = "immediate"
            self.uwb_online_initialisation.params['z_score_threshold'] = 3
            self.uwb_online_initialisation.params['outlier_count_threshold'] = 3

        else:
            self.uwb_online_initialisation.params['outlier_removing'] = "None"
            
        if setting == "linear_no_bias":

            self.uwb_online_initialisation.params['rough_estimate_method'] = "simple_linear"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            
            
        elif setting == "linear_constant_bias":

            self.uwb_online_initialisation.params['rough_estimate_method'] = "simple_linear"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = True
            

        elif setting == "linear_linear_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "simple_linear"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            

        elif setting == "linear_both_biases":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "simple_linear"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = True
            



        elif setting == "reweighted_linear_no_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            

        elif setting == "reweighted_linear_constant_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = True
            

        elif setting == "reweighted_linear_linear_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            

        elif setting == "reweighted_linear_both_biases":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = True
            



        elif setting == "filtered_linear_no_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "simple_linear"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            

        elif setting == "filtered_linear_constant_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "simple_linear"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = True
            

        elif setting == "filtered_linear_linear_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "simple_linear"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            

        elif setting == "filtered_linear_both_biases":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "simple_linear"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = True
            



        elif setting == "filtered_reweighted_linear_no_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            

        elif setting == "filtered_reweighted_linear_constant_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = True
            

        elif setting == "filtered_reweighted_linear_linear_bias":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            

        elif setting == "filtered_reweighted_linear_both_biases":
            
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = True

    
        elif setting == "regularised_weighted_linear_no_bias":
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            self.uwb_online_initialisation.params['regularise'] = True

        elif setting == "regularised_weighted_linear_constant_bias":
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = False
            self.uwb_online_initialisation.params['use_constant_bias'] = True
            self.uwb_online_initialisation.params['regularise'] = True
        
        elif setting == "regularised_weighted_linear_linear_bias":
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = False
            self.uwb_online_initialisation.params['regularise'] = True

        elif setting == "regularised_weighted_linear_both_biases":
            self.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted"
            self.uwb_online_initialisation.params['use_linear_bias'] = True
            self.uwb_online_initialisation.params['use_constant_bias'] = True
            self.uwb_online_initialisation.params['regularise'] = True

        else:
            pass

    def waypoint_checkpoint(self, drone_position):
        """Function to check if the drone is close to a waypoint and if so, update the remaining waypoints."""
        if self.uwb_online_initialisation.remaining_waypoints:
            if np.linalg.norm(np.array(drone_position) - np.array(self.uwb_online_initialisation.remaining_waypoints[0])) < 0.01:
                self.uwb_online_initialisation.passed_waypoints.append(self.uwb_online_initialisation.remaining_waypoints[0])
                self.uwb_online_initialisation.remaining_waypoints.pop(0)

                last_waypoint = self.uwb_online_initialisation.passed_waypoints[-1]

                if last_waypoint in self.uwb_online_initialisation.current_optimal_waypoints:
                    self.uwb_online_initialisation.current_optimal_waypoints.remove(last_waypoint)
                    
                print(f"Waypoint reached, {len(self.uwb_online_initialisation.remaining_waypoints)} remaining")

    def run_pipeline_linear(self, measurement_vector=None, anchor_id="1"):

        """Single anchor pipeline"""
        iteration = -1
        anchor = next(anchor for anchor in self.drone_sim.unknown_anchors if anchor.anchor_ID == anchor_id)


        while True:
            iteration += 1

            if measurement_vector is None:
                if iteration >= len(self.drone_sim.drone_trajectory.spline_x):
                    break
                
                if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] == "optimised_trajectory":
                    break # This is the pre-optimisation part of the pipeline

                drone_x, drone_y, drone_z = self.drone_sim.drone_trajectory.spline_x[iteration], self.drone_sim.drone_trajectory.spline_y[iteration], self.drone_sim.drone_trajectory.spline_z[iteration]

                self.uwb_online_initialisation.drone_postion = [drone_x, drone_y, drone_z]

                
                distance = anchor.request_distance(drone_x, drone_y, drone_z)
                self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], distance, anchor_id)


            # Case where the measurement vector is provided
            else:
                # print("Iteration", iteration)
                # print("Measurement vector", len(measurement_vector))
                if iteration >= len(measurement_vector):
                    break

                if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] == "optimised_trajectory":
                    break # This is the pre-optimisation part of the pipeline

                drone_x, drone_y, drone_z, distance = measurement_vector[iteration]

                self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], distance, anchor_id)

            
            
            # Compute the stopping criterion variables
            FIM = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["FIM"]
            GDOP = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["GDOP"]
            mean_residuals = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["residuals"]
            condition_number = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["condition_number"]
            covariances = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["covariances"]
            weights = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["linear_ls_weights"]
            verification_value = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["verification_vector"]
            position_delta = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["consecutive_distances_vector"]


            
            anchor_index = next(index for index, anchor in enumerate(self.drone_sim.unknown_anchors) if anchor.anchor_ID == anchor_id)

            error = self.uwb_online_initialisation.calculate_position_error(self.drone_sim.unknown_anchors[anchor_index].get_anchor_coordinates(), self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][:3])
            constant_bias_error = np.linalg.norm(self.drone_sim.unknown_anchors[0].bias - self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][3])
            linear_bias_error = np.linalg.norm(self.drone_sim.unknown_anchors[0].linear_bias - self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][4])

            # Store the metrics in the vectors of the class for plotting or writing to a csv file
            self.gdop_vector[anchor_id] = GDOP
            self.fim_vector[anchor_id] = FIM
            self.condition_number_vector[anchor_id] = condition_number
            self.covariances_vector[anchor_id] = covariances
            self.residuals_vector[anchor_id] = mean_residuals
            self.weight_vector[anchor_id] = weights
            self.verifications_vector[anchor_id] = verification_value
            self.pos_delta_vector[anchor_id] = position_delta

            # To stay consistent in the length of the vectors, append a default value
            if self.number_of_measurements[anchor_id] < len(condition_number):
                self.error_vector[anchor_id].append(error)
                self.constant_bias_error_vector[anchor_id].append(constant_bias_error)
                self.linear_bias_error_vector[anchor_id].append(linear_bias_error)
                self.constant_bias_vector[anchor_id].append(self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][3])
                self.linear_bias_vector[anchor_id].append(self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][4])
                
                # self.measurement_vector[anchor_id].append([drone_x, drone_y, drone_z, distance])
                self.number_of_measurements[anchor_id] += 1

        # check stopping criterion ?



    def run_pipeline_pre_optimisation(self, measurement_vector=None, anchor_id="1"):
        """Single anchor pipeline"""
        iteration = -1
        anchor_ID = anchor_id

        anchor = next(anchor for anchor in self.drone_sim.unknown_anchors if anchor.anchor_ID == anchor_id)


        while True:
            iteration += 1

            if measurement_vector is None:
                if iteration >= len(self.drone_sim.drone_trajectory.spline_x):
                    break
                
                if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] == "optimised_trajectory":
                    break # This is the pre-optimisation part of the pipeline

                drone_x, drone_y, drone_z = self.drone_sim.drone_trajectory.spline_x[iteration], self.drone_sim.drone_trajectory.spline_y[iteration], self.drone_sim.drone_trajectory.spline_z[iteration]

                self.uwb_online_initialisation.drone_postion = [drone_x, drone_y, drone_z]

                
                distance = anchor.request_distance(drone_x, drone_y, drone_z)
                self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], distance, anchor_id)


            # Case where the measurement vector is provided
            else:
                if iteration >= len(measurement_vector):
                    break

                if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] == "optimised_trajectory":
                    break # This is the pre-optimisation part of the pipeline

                drone_x, drone_y, drone_z, distance = measurement_vector[iteration]

                self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], distance, anchor_id)

            
            
            # Compute the stopping criterion variables
            FIM = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["FIM"]
            GDOP = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["GDOP"]
            mean_residuals = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["residuals"]
            condition_number = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["condition_number"]
            covariances = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["covariances"]
            weights = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["linear_ls_weights"]
            verification_value = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["verification_vector"]
            position_delta = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["consecutive_distances_vector"]


            
            anchor_index = next(index for index, anchor in enumerate(self.drone_sim.unknown_anchors) if anchor.anchor_ID == anchor_id)

            error = self.uwb_online_initialisation.calculate_position_error(self.drone_sim.unknown_anchors[anchor_index].get_anchor_coordinates(), self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][:3])
            constant_bias_error = np.linalg.norm(self.drone_sim.unknown_anchors[0].bias - self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][3])
            linear_bias_error = np.linalg.norm(self.drone_sim.unknown_anchors[0].linear_bias - self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][4])

            # Store the metrics in the vectors of the class for plotting or writing to a csv file
            self.gdop_vector[anchor_id] = GDOP
            self.fim_vector[anchor_id] = FIM
            self.condition_number_vector[anchor_id] = condition_number
            self.covariances_vector[anchor_id] = covariances
            self.residuals_vector[anchor_id] = mean_residuals
            self.weight_vector[anchor_id] = weights
            self.verifications_vector[anchor_id] = verification_value
            self.pos_delta_vector[anchor_id] = position_delta

            # To stay consistent in the length of the vectors, append a default value
            if self.number_of_measurements[anchor_id] < len(condition_number):
                self.error_vector[anchor_id].append(error)
                self.constant_bias_error_vector[anchor_id].append(constant_bias_error)
                self.linear_bias_error_vector[anchor_id].append(linear_bias_error)
                self.constant_bias_vector[anchor_id].append(self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][3])
                self.linear_bias_vector[anchor_id].append(self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][4])
                
                self.measurement_vector[anchor_id].append([drone_x, drone_y, drone_z, distance])
                self.number_of_measurements[anchor_id] += 1


        # At the end of the while loop, the rough estimate is done but in the case of no stopping criterion, we do not have a refined estimate
        if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] != "optimised_trajectory":
            
            anchor_measurement_dictionary = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]
            measurements = []
            for distance, position in zip(anchor_measurement_dictionary["distances_pre_rough_estimate"]+anchor_measurement_dictionary["distances_post_rough_estimate"], anchor_measurement_dictionary["positions_pre_rough_estimate"] + anchor_measurement_dictionary["positions_post_rough_estimate"]):
                x, y, z = position
                measurements.append([x, y, z, distance])

            estimator, covariance_matrix = self.uwb_online_initialisation.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess=anchor_measurement_dictionary["estimator_rough_linear"])

            # Update the dictionnary with the non-linear refined rough estimate
            anchor_measurement_dictionary["estimator_rough_non_linear"] = estimator
            anchor_measurement_dictionary["estimator"] = estimator

        # From here the initialisation is finished, the non_linear method is done but the anchor might not be in the optimised trajectory phase as the stopping criterion is not necessarily met

    def run_pipeline_post_optimisation(self, measurement_vector, anchor_id="1"):
        if measurement_vector is None:
            optimal_waypoints = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["optimal_waypoints"]
            if len(optimal_waypoints) > 2:
                optimal_trajectory = Trajectory()
                optimal_waypoint_objects = [Waypoint(*waypoint) for waypoint in optimal_waypoints]
                optimal_trajectory.construct_trajectory_linear(optimal_waypoint_objects)
                anchor_index = next(index for index, anchor in enumerate(self.drone_sim.unknown_anchors) if anchor.anchor_ID == anchor_id)
                measurement_vector = [[x, y, z, self.drone_sim.unknown_anchors[anchor_index].request_distance(x, y, z)] for x, y, z in zip(optimal_trajectory.spline_x, optimal_trajectory.spline_y, optimal_trajectory.spline_z)]
            else:
                print("The optimal trajectory is not long enough to gather measurements")
                return
        for meas in measurement_vector:

            x,y,z = meas[:3]
            self.waypoint_checkpoint([x, y, z])
            if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] == "initialised":
                print("ERROR: The anchor is initialised")
                break
            elif self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] != "optimised_trajectory":
                print("ERROR: The anchor is not in the optimised trajectory phase, this can lead to errors in the pipeline")
                break
            else:
                drone_x, drone_y, drone_z, distance = meas
                self.uwb_online_initialisation.drone_postion = [drone_x, drone_y, drone_z]

                self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], distance, anchor_id)
        
        # end of the loop, refine the estimate with the non-linear method a last time 
        anchor_measurement_dictionary = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]
        measurements = []
        for distance, position in zip(anchor_measurement_dictionary["distances_pre_rough_estimate"]+anchor_measurement_dictionary["distances_post_rough_estimate"], anchor_measurement_dictionary["positions_pre_rough_estimate"] + anchor_measurement_dictionary["positions_post_rough_estimate"]):
            x, y, z = position
            measurements.append([x, y, z, distance])

        estimator, covariance_matrix = self.uwb_online_initialisation.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess=anchor_measurement_dictionary["estimator_rough_non_linear"])
        anchor_measurement_dictionary["estimator"] = estimator



    def run_pipeline_full(self, measurement_vector=None, anchor_id="1"):

        """Single anchor pipeline"""
        iteration = -1

        anchor = next(anchor for anchor in self.drone_sim.unknown_anchors if anchor.anchor_ID == anchor_id)


        while True:
            iteration += 1

            if measurement_vector is None:
                if iteration >= len(self.drone_sim.drone_trajectory.spline_x):
                    break
                
                if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] == "optimised_trajectory":
                    break # This is the pre-optimisation part of the pipeline

                drone_x, drone_y, drone_z = self.drone_sim.drone_trajectory.spline_x[iteration], self.drone_sim.drone_trajectory.spline_y[iteration], self.drone_sim.drone_trajectory.spline_z[iteration]

                self.uwb_online_initialisation.drone_postion = [drone_x, drone_y, drone_z]

                
                distance = anchor.request_distance(drone_x, drone_y, drone_z)
                self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], distance, anchor_id)


            # Case where the measurement vector is provided
            else:
                if iteration >= len(measurement_vector):
                    break

                if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] == "optimised_trajectory":
                    break # This is the pre-optimisation part of the pipeline

                drone_x, drone_y, drone_z, distance = measurement_vector[iteration]

                self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], distance, anchor_id)

            
            
            # Compute the stopping criterion variables
            FIM = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["FIM"]
            GDOP = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["GDOP"]
            mean_residuals = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["residuals"]
            condition_number = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["condition_number"]
            covariances = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["covariances"]
            weights = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["linear_ls_weights"]
            verification_value = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["verification_vector"]
            position_delta = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["consecutive_distances_vector"]


            
            anchor_index = next(index for index, anchor in enumerate(self.drone_sim.unknown_anchors) if anchor.anchor_ID == anchor_id)

            error = self.uwb_online_initialisation.calculate_position_error(self.drone_sim.unknown_anchors[anchor_index].get_anchor_coordinates(), self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][:3])
            constant_bias_error = np.linalg.norm(self.drone_sim.unknown_anchors[0].bias - self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][3])
            linear_bias_error = np.linalg.norm(self.drone_sim.unknown_anchors[0].linear_bias - self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][4])

            # Store the metrics in the vectors of the class for plotting or writing to a csv file
            self.gdop_vector[anchor_id] = GDOP
            self.fim_vector[anchor_id] = FIM
            self.condition_number_vector[anchor_id] = condition_number
            self.covariances_vector[anchor_id] = covariances
            self.residuals_vector[anchor_id] = mean_residuals
            self.weight_vector[anchor_id] = weights
            self.verifications_vector[anchor_id] = verification_value
            self.pos_delta_vector[anchor_id] = position_delta

            # To stay consistent in the length of the vectors, append a default value
            if self.number_of_measurements[anchor_id] < len(condition_number):
                self.error_vector[anchor_id].append(error)
                self.constant_bias_error_vector[anchor_id].append(constant_bias_error)
                self.linear_bias_error_vector[anchor_id].append(linear_bias_error)
                self.constant_bias_vector[anchor_id].append(self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][3])
                self.linear_bias_vector[anchor_id].append(self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["estimator_rough_linear"][4])
                
                self.measurement_vector[anchor_id].append([drone_x, drone_y, drone_z, distance])
                self.number_of_measurements[anchor_id] += 1


        # At the end of the while loop, the rough estimate is done but in the case of no stopping criterion, we do not have a refined estimate
        if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] != "optimised_trajectory":
            
            anchor_measurement_dictionary = self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]
            measurements = []
            for distance, position in zip(anchor_measurement_dictionary["distances_pre_rough_estimate"]+anchor_measurement_dictionary["distances_post_rough_estimate"], anchor_measurement_dictionary["positions_pre_rough_estimate"] + anchor_measurement_dictionary["positions_post_rough_estimate"]):
                x, y, z = position
                measurements.append([x, y, z, distance])

            estimator, covariance_matrix = self.uwb_online_initialisation.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess=anchor_measurement_dictionary["estimator_rough_linear"])

            # Update the dictionnary with the non-linear refined rough estimate
            anchor_measurement_dictionary["estimator_rough_non_linear"] = estimator
            anchor_measurement_dictionary["estimator"] = estimator

            # Run the trajectory optimisation
            anchor_estimate_variance = anchor_measurement_dictionary["covariance_matrix_rough_linear"][:3]

            previous_measurement_positions = anchor_measurement_dictionary["positions_pre_rough_estimate"]

            initial_remaining_waypoints = self.uwb_online_initialisation.remaining_waypoints
            
            self.uwb_online_initialisation.trajectory_optimiser.method = self.uwb_online_initialisation.params["trajectory_optimisation_method"]
            link_method = self.uwb_online_initialisation.params["link_method"]
            drone_position = previous_measurement_positions[-1]
            # Optimize the trajectory using the previous measurements and the rough estimate of the anchor 
            optimal_waypoints = self.uwb_online_initialisation.trajectory_optimiser.optimize_waypoints_incrementally_spherical(drone_position, estimator, anchor_estimate_variance, previous_measurement_positions, initial_remaining_waypoints, radius_of_search = 0.4, max_waypoints=10, marginal_gain_threshold=0.01)
            self.uwb_online_initialisation.current_optimal_waypoints = optimal_waypoints
            drone_position = self.uwb_online_initialisation.drone_position
            full_waypoints, optimal_waypoints, link_waypoints, mission_waypoints = self.uwb_online_initialisation.trajectory_optimiser.compute_new_mission_waypoints(drone_position, initial_remaining_waypoints, optimal_waypoints, link_method)
            
            # Update the status of the anchor to optimised trajectory
            self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] = "optimised_trajectory"

            # Update the status of the drone to on optimal trajectory
            self.uwb_online_initialisation.status = "on_optimal_trajectory"





        full_waypoints = self.uwb_online_initialisation.current_optimal_waypoints + self.uwb_online_initialisation.current_link_waypoints + self.uwb_online_initialisation.remaining_waypoints
        full_waypoints = [Waypoint(*waypoint) for waypoint in full_waypoints]

        optimal_trajectory = Trajectory(0.1, 0.05)
        optimal_trajectory.construct_trajectory_linear(full_waypoints)

        self.optimal_trajectories[anchor_id] = optimal_trajectory

        # Gather measurements on the optimal trajectory
        optimal_x, optimal_y, optimal_z = self.optimal_trajectories[anchor_id].spline_x, self.optimal_trajectories[anchor_id].spline_y, self.optimal_trajectories[anchor_id].spline_z

        anchor = next(anchor for anchor in self.drone_sim.unknown_anchors if anchor.anchor_ID == anchor_id)

        for drone_x, drone_y, drone_z in zip(optimal_x, optimal_y, optimal_z):
            anchor_id = next(iter(self.uwb_online_initialisation.anchor_measurements_dictionary))

            
            if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["status"] == "initialised":
                break
            else:
                self.uwb_online_initialisation.drone_postion = [drone_x, drone_y, drone_z]
                
                distance = anchor.request_distance(drone_x, drone_y, drone_z)
                self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], distance, anchor_id)




    def run_pipeline_multiple_anchors(self, measurement_vector=None, anchor_ids=["1", "2", "3", "4"]):

        drone_trajectory = copy.deepcopy(self.drone_sim.drone_trajectory)
        self.drone_trajectory = drone_trajectory

        self.uwb_online_initialisation.remaining_waypoints = [waypoint.get_coordinates() for waypoint in drone_trajectory.waypoints]
        print(self.uwb_online_initialisation.remaining_waypoints)
        iteration = -1
        while True:
            iteration += 1

            if measurement_vector is None:
                if iteration >= len(drone_trajectory.spline_x):
                    print("Warning: The trajectory is finished - some anchors might not be fully optimised")
                    break

                drone_x, drone_y, drone_z = drone_trajectory.spline_x[iteration], drone_trajectory.spline_y[iteration], drone_trajectory.spline_z[iteration]
                self.waypoint_checkpoint([drone_x, drone_y, drone_z])

                for anchor_id in anchor_ids:
                    anchor = next(anchor for anchor in self.drone_sim.unknown_anchors if anchor.anchor_ID == anchor_id)
                    distance = anchor.request_distance(drone_x, drone_y, drone_z)
                    optimal_waypoints = self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], distance, anchor_id)

                    if optimal_waypoints is not None and len(optimal_waypoints) > 1:    
                        self.optimal_waypoints[anchor_id] = copy.deepcopy(optimal_waypoints)

                        # update the drone trajectory
                        drone_trajectory = Trajectory(1, 0.05)
                        optimal_waypoints = [Waypoint(*waypoint) for waypoint in optimal_waypoints]
                        # insert current drone position as a waypoint in front of the others:
                        optimal_waypoints.insert(0, Waypoint(drone_x, drone_y, drone_z))
                        drone_trajectory.construct_trajectory_linear(optimal_waypoints)

                        self.optimal_trajectories[anchor_id] = drone_trajectory
                        

                        iteration = -1
                anchors_initialised = [anchor.anchor_ID for anchor in self.drone_sim.unknown_anchors if self.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["status"] == "initialised"]
                # for anchor in self.drone_sim.unknown_anchors:
                #     print(f"Anchor {anchor.anchor_ID} is in status {self.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]['status']}")

                # print(self.uwb_online_initialisation.status)
                if len(anchors_initialised) == len(anchor_ids):
                    print("All anchors are initialised")
                    print("Final drone position", [drone_x, drone_y, drone_z])
                    break

            else:
                print("This function does not support the measurement vector yet")
                pass
                




    # For analysis puposes
    def compute_angular_span_around_point(self, vectors, point):
        """Utility function to compute the angular span around a given point.
        
        Parameters:
        vectors (list): List of vectors in the form (x, y, z).
        point (list): The point around which to compute the angular span in the form (x0, y0, z0).
        
        Returns:
        azimuthal_span (float): The azimuthal span around the point.
        elevation_span (float): The elevation span around the point.
        """

        def cartesian_to_spherical(vector, point):
            # Translate the vector so that 'point' is the new origin
            translated_vector = (vector[0] - point[0], vector[1] - point[1], vector[2] - point[2])
            x, y, z = translated_vector
            
            # Compute spherical coordinates
            r = np.sqrt(x**2 + y**2 + z**2)  # Radial distance (not used directly here)
            azimuthal_angle = np.arctan2(y, x)  # Azimuthal angle (phi)
            elevation_angle = np.arctan2(z, np.sqrt(x**2 + y**2))  # Elevation angle (theta)
            
            return azimuthal_angle, elevation_angle
        
        def angular_differences(angles):
            # Compute differences between consecutive angles
            diffs = np.diff(angles, append=angles[0] + (angles[-1] - angles[0]))  # Wrap around
            diffs = np.where(diffs > np.pi, diffs - 2 * np.pi, diffs)  # Handle positive wrap-around
            diffs = np.where(diffs < -np.pi, diffs + 2 * np.pi, diffs)  # Handle negative wrap-around
            return diffs
        
        def kadane_algorithm(arr):
            # Find maximum sum subarray (positive span)
            max_ending_here = max_so_far = arr[0]
            for x in arr[1:]:
                max_ending_here = max(x, max_ending_here + x)
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        def min_kadane_algorithm(arr):
            # Find minimum sum subarray (negative span)
            min_ending_here = min_so_far = arr[0]
            for x in arr[1:]:
                min_ending_here = min(x, min_ending_here + x)
                min_so_far = min(min_so_far, min_ending_here)
            return min_so_far

        # Convert vectors to spherical coordinates
        spherical_coords = [cartesian_to_spherical(vector, point) for vector in vectors]
        
        azimuthal_angles = [angle[0] for angle in spherical_coords]
        elevation_angles = [angle[1] for angle in spherical_coords]
        
        # Compute angular differences for azimuthal and elevation angles
        azimuthal_diffs = angular_differences(azimuthal_angles)
        elevation_diffs = angular_differences(elevation_angles)
        
        # Compute the maximum and minimum angular spans using Kadane’s algorithms
        max_azimuthal_span = kadane_algorithm(azimuthal_diffs)
        min_azimuthal_span = min_kadane_algorithm(azimuthal_diffs)
        max_elevation_span = kadane_algorithm(elevation_diffs)
        min_elevation_span = min_kadane_algorithm(elevation_diffs)
        
        # The largest span is the maximum of absolute values of max and min spans
        azimuthal_span = max(abs(max_azimuthal_span), abs(min_azimuthal_span))
        elevation_span = max(abs(max_elevation_span), abs(min_elevation_span))
        
        azimuthal_span_degrees = azimuthal_span * 180 / np.pi
        elevation_span_degrees = elevation_span * 180 / np.pi

        return azimuthal_span_degrees, elevation_span_degrees



    # Helper function for csv writing and reading
    def write_measurements_to_csv(self, path, anchor_id):
        """Function to create some data with measurements and write it to a CSV file."""
        # Open a CSV file in append mode

        anchor = next(anchor for anchor in self.drone_sim.unknown_anchors if anchor.anchor_ID == anchor_id)

        row = [str(anchor.get_anchor_coordinates()), str(self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["positions_pre_rough_estimate"]), str(self.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["distances_pre_rough_estimate"])]
        
        # Open a CSV file in append mode
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def load_measurement_data_from_csv(self, path):

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

    def save_row_to_csv(self, csv_file, row):
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)



















def run_simulation_linear_least_squares_comparison():
    """Run the simulation to gather measurements and compute a linear least squares estimation for the unknown anchor, using different bias settings.
    A stopping criterion is used to extract a subset of the measurements, and the results are stored in a CSV file."""



    settings1 = ["linear_no_bias", "linear_constant_bias", "linear_linear_bias", "linear_both_biases"]
    settings2 = ["reweighted_linear_no_bias", "reweighted_linear_constant_bias", "reweighted_linear_linear_bias", "reweighted_linear_both_biases"]
    settings3 = ["filtered_linear_no_bias", "filtered_linear_constant_bias", "filtered_linear_linear_bias", "filtered_linear_both_biases"]
    settings4 = ["filtered_reweighted_linear_no_bias", "filtered_reweighted_linear_constant_bias", "filtered_reweighted_linear_linear_bias", "filtered_reweighted_linear_both_biases"]
    settings = settings1 + settings2 + settings3 + settings4
    
    # Iterate over environments
    for environment in range(100000):
        print(f"Environment {environment}")
        calculate_online_initialisation = CalculateOnlineInitialisation()
        calculate_online_initialisation.uwb_online_initialisation.params["stopping_criteria"] = None
        calculate_online_initialisation.randomise_environment()

        unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]

        calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)

        trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
        calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy
        
        

        # Calculate the measurement vector to use for the simulation (Needed to compare methods on the same data)
        # measurement_vector = None
        calculate_online_initialisation.gather_measurements()
        measurement_vector = calculate_online_initialisation.measurement_vector[unknown_anchor.anchor_ID]

        gdop = calculate_online_initialisation.uwb_online_initialisation.compute_GDOP(measurements=measurement_vector, target_coords=unknown_anchor.get_anchor_coordinates())

        if gdop > 4:
            continue

        errors = []
        need_to_exit = False
        measurement_number = []
        # If needed to compare, start the loop here 
        for setting in settings:
            # Set parameters for the simulation here
            calculate_online_initialisation.write_settings(setting)

            # Run the simulation
            try:
                measurement_vector_copy = copy.deepcopy(measurement_vector)
                calculate_online_initialisation.run_pipeline_linear(measurement_vector_copy, unknown_anchor.anchor_ID)
                
            except:
                print(f"Error in the simulation for setting {setting}")
                need_to_exit = True
                break
            # Save the error values / other metrics before the next iteration with different parameters
            error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(unknown_anchor.get_anchor_coordinates(), calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][:3])
            error_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(unknown_anchor.get_anchor_gt_estimator(), calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"])
            constant_bias_error = np.linalg.norm(unknown_anchor.bias - calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][3])
            linear_bias_error = np.linalg.norm(unknown_anchor.linear_bias - calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][4])
            non_linear_error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(unknown_anchor.get_anchor_coordinates(), calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"][:3])
            errors.extend([error, error_full, constant_bias_error, linear_bias_error, non_linear_error])
            
            measurement_number.append(len(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]))

            calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
            
        if need_to_exit:
            continue
        
        
        # Calculate metrics to store in the CSV
        number_of_measurements = measurement_number 
        min_angle_between_two_consecutive_measurements = np.rad2deg(np.atan(calculate_online_initialisation.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold']))

        anchor_position_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
        anchor_distances_gt = np.linalg.norm([np.array(row[:3]) - np.array(anchor_position_gt) for row in measurement_vector] , axis=1)

        measured_noise_mean = np.mean(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))
        measured_noise_var = np.var(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))

        min_distance_to_anchor = np.min(anchor_distances_gt)
        max_distance_to_anchor = np.max(anchor_distances_gt)
        mean_distance_to_anchor = np.mean(anchor_distances_gt)
        std_distance_to_anchor = np.std(anchor_distances_gt)

        angular_span_azimuth, angular_span_elevation = calculate_online_initialisation.compute_angular_span_around_point([row[:3] for row in measurement_vector], anchor_position_gt)

        constant_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].bias
        linear_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias

        noise_variance_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance
        outlier_probability = calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability


        # Prepare the rows to write to the CSV file
        environment_settings = [number_of_measurements, min_angle_between_two_consecutive_measurements, min_distance_to_anchor, max_distance_to_anchor, mean_distance_to_anchor, std_distance_to_anchor, angular_span_elevation, angular_span_azimuth, constant_bias_gt, linear_bias_gt, noise_variance_gt, measured_noise_var, measured_noise_mean, outlier_probability]
        
        row = environment_settings + errors
        
        csv_file = csv_dir / 'linear_least_squares.csv'
        calculate_online_initialisation.save_row_to_csv(csv_file, row)

def run_simulation_regularisation():
    """Run the simulation to gather measurements and compute a linear least squares estimation for the unknown anchor, using different bias settings.
    A stopping criterion is used to extract a subset of the measurements, and the results are stored in a CSV file."""


    settings = ["reweighted_linear_no_bias", "reweighted_linear_constant_bias", "reweighted_linear_linear_bias", "reweighted_linear_both_biases", "regularised_weighted_linear_no_bias", "regularised_weighted_linear_constant_bias", "regularised_weighted_linear_linear_bias", "regularised_weighted_linear_both_biases"]

    
    # Iterate over environments
    for environment in range(100000):
        print(f"Environment {environment}")
        calculate_online_initialisation = CalculateOnlineInitialisation()
        calculate_online_initialisation.uwb_online_initialisation.params["stopping_criteria"] = None
        calculate_online_initialisation.randomise_environment()

        unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]

        calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)

        trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
        calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy
        
        

        # Calculate the measurement vector to use for the simulation (Needed to compare methods on the same data)
        # measurement_vector = None
        calculate_online_initialisation.gather_measurements()
        measurement_vector = calculate_online_initialisation.measurement_vector[unknown_anchor.anchor_ID]
        length = len(measurement_vector)
        measurement_vector = measurement_vector[:int(length//1.2)]
        gdop = calculate_online_initialisation.uwb_online_initialisation.compute_GDOP(measurements=measurement_vector, target_coords=unknown_anchor.get_anchor_coordinates())

        if gdop > 4:
            continue

        errors = []
        need_to_exit = False
        measurement_number = []
        # If needed to compare, start the loop here 
        for setting in settings:
            # Set parameters for the simulation here
            calculate_online_initialisation.write_settings(setting)

            # Run the simulation
            try:
                measurement_vector_copy = copy.deepcopy(measurement_vector)
                calculate_online_initialisation.run_pipeline_linear(measurement_vector_copy, unknown_anchor.anchor_ID)
                
            except:
                print(f"Error in the simulation for setting {setting}")
                need_to_exit = True
                break
            # Save the error values / other metrics before the next iteration with different parameters
            error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(unknown_anchor.get_anchor_coordinates(), calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][:3])
            error_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(unknown_anchor.get_anchor_gt_estimator(), calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"])
            constant_bias_error = np.linalg.norm(unknown_anchor.bias - calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][3])
            linear_bias_error = np.linalg.norm(unknown_anchor.linear_bias - calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][4])
            non_linear_error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(unknown_anchor.get_anchor_coordinates(), calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"][:3])
            errors.extend([error, error_full, constant_bias_error, linear_bias_error, non_linear_error])
            
            measurement_number.append(len(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]))

            calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
            
        if need_to_exit:
            continue
        
        
        # Calculate metrics to store in the CSV
        number_of_measurements = measurement_number 
        min_angle_between_two_consecutive_measurements = np.rad2deg(np.atan(calculate_online_initialisation.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold']))

        anchor_position_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
        anchor_distances_gt = np.linalg.norm([np.array(row[:3]) - np.array(anchor_position_gt) for row in measurement_vector] , axis=1)

        measured_noise_mean = np.mean(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))
        measured_noise_var = np.var(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))

        min_distance_to_anchor = np.min(anchor_distances_gt)
        max_distance_to_anchor = np.max(anchor_distances_gt)
        mean_distance_to_anchor = np.mean(anchor_distances_gt)
        std_distance_to_anchor = np.std(anchor_distances_gt)

        angular_span_azimuth, angular_span_elevation = calculate_online_initialisation.compute_angular_span_around_point([row[:3] for row in measurement_vector], anchor_position_gt)

        constant_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].bias
        linear_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias

        noise_variance_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance
        outlier_probability = calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability


        # Prepare the rows to write to the CSV file
        environment_settings = [number_of_measurements, min_angle_between_two_consecutive_measurements, min_distance_to_anchor, max_distance_to_anchor, mean_distance_to_anchor, std_distance_to_anchor, angular_span_elevation, angular_span_azimuth, constant_bias_gt, linear_bias_gt, noise_variance_gt, measured_noise_var, measured_noise_mean, outlier_probability]
        
        row = environment_settings + errors
        
        csv_file = csv_dir / 'regularisation_errors.csv'
        calculate_online_initialisation.save_row_to_csv(csv_file, row)

def run_simulation_reweighted_ls():
    settings = ["regularised_weighted_linear_constant_bias"]

    
    # Iterate over environments
    for environment in range(100000):
        print(f"Environment {environment}")
        calculate_online_initialisation = CalculateOnlineInitialisation()
        calculate_online_initialisation.uwb_online_initialisation.params["stopping_criteria"] = None
        calculate_online_initialisation.randomise_environment()

        unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]

        calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)

        trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
        calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy
        
        

        # Calculate the measurement vector to use for the simulation (Needed to compare methods on the same data)
        # measurement_vector = None
        calculate_online_initialisation.gather_measurements()
        measurement_vector = calculate_online_initialisation.measurement_vector[unknown_anchor.anchor_ID]
        length = len(measurement_vector)
        measurement_vector = measurement_vector[:int(length//1)]
        gdop = calculate_online_initialisation.uwb_online_initialisation.compute_GDOP(measurements=measurement_vector, target_coords=unknown_anchor.get_anchor_coordinates())

        if gdop > 3:
            continue

        errors = []
        need_to_exit = False
        measurement_number = []
        # If needed to compare, start the loop here 
        for setting in settings:
            # Set parameters for the simulation here
            calculate_online_initialisation.write_settings(setting)

            # Run the simulation
            try:
                measurement_vector_copy = copy.deepcopy(measurement_vector)
                calculate_online_initialisation.run_pipeline_linear(measurement_vector_copy, unknown_anchor.anchor_ID)
                
            except:
                print(f"Error in the simulation for setting {setting}")
                need_to_exit = True
                break
            # Save the error values / other metrics before the next iteration with different parameters
            error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(unknown_anchor.get_anchor_coordinates(), calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][:3])
            error_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(unknown_anchor.get_anchor_gt_estimator(), calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"])
            constant_bias_error = np.linalg.norm(unknown_anchor.bias - calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][3])
            linear_bias_error = np.linalg.norm(unknown_anchor.linear_bias - calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][4])
            non_linear_error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(unknown_anchor.get_anchor_coordinates(), calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"][:3])
            errors.extend([error, error_full, constant_bias_error, linear_bias_error, non_linear_error])
            
            measurement_number.append(len(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]))

            calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
            
        if need_to_exit:
            continue
        
        
        # Calculate metrics to store in the CSV
        number_of_measurements = measurement_number 
        min_angle_between_two_consecutive_measurements = np.rad2deg(np.atan(calculate_online_initialisation.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold']))

        anchor_position_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
        anchor_distances_gt = np.linalg.norm([np.array(row[:3]) - np.array(anchor_position_gt) for row in measurement_vector] , axis=1)

        measured_noise_mean = np.mean(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))
        measured_noise_var = np.var(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))

        min_distance_to_anchor = np.min(anchor_distances_gt)
        max_distance_to_anchor = np.max(anchor_distances_gt)
        mean_distance_to_anchor = np.mean(anchor_distances_gt)
        std_distance_to_anchor = np.std(anchor_distances_gt)

        angular_span_azimuth, angular_span_elevation = calculate_online_initialisation.compute_angular_span_around_point([row[:3] for row in measurement_vector], anchor_position_gt)

        constant_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].bias
        linear_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias

        noise_variance_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance
        outlier_probability = calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability


        # Prepare the rows to write to the CSV file
        environment_settings = [number_of_measurements, min_angle_between_two_consecutive_measurements, min_distance_to_anchor, max_distance_to_anchor, mean_distance_to_anchor, std_distance_to_anchor, angular_span_elevation, angular_span_azimuth, constant_bias_gt, linear_bias_gt, noise_variance_gt, measured_noise_var, measured_noise_mean, outlier_probability]
        
        row = environment_settings + errors
        
        csv_file = csv_dir / 'reweighted_ls.csv'
        calculate_online_initialisation.save_row_to_csv(csv_file, row)

def run_simulation_stopping_metrics_comparison():
    """Run the simulation to gather measurements and metrics for the unknown anchor, using all space and all measurement available. (no stopping criterion)
    The metrics are stored in a CSV file, ready to be analysed.
    
    GOAL: choose which metric is the more reliable and gives the more information to get the best estimation of the anchor position with minimal error.""" 

    # Define the path of the csv to store the data
    path = csv_dir / 'metrics.csv'

    # iterate over environments
    for environment in range(1000):
        calculate_online_initialisation = CalculateOnlineInitialisation()
        calculate_online_initialisation.randomise_environment()

        unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]
        calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
        calculate_online_initialisation.uwb_online_initialisation.trajectory = calculate_online_initialisation.drone_sim.drone_trajectory

        calculate_online_initialisation.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted" # Method to use for the rough estimate, either simple_linear or linear_reweighted

        calculate_online_initialisation.gather_measurements()
        measurement_vector = calculate_online_initialisation.measurement_vector[unknown_anchor.anchor_ID]
        length = len(measurement_vector)
        measurement_vector = measurement_vector[:int(length//(2/4))]

        measurement_vector_copy = copy.deepcopy(measurement_vector)

        calculate_online_initialisation.run_pipeline_linear(measurement_vector_copy, unknown_anchor.anchor_ID)

        row = [str(calculate_online_initialisation.gdop_vector[unknown_anchor.anchor_ID]), str([np.linalg.det(row) for row in calculate_online_initialisation.fim_vector[unknown_anchor.anchor_ID]]), str(calculate_online_initialisation.condition_number_vector[unknown_anchor.anchor_ID]), str(calculate_online_initialisation.residuals_vector[unknown_anchor.anchor_ID]), str([np.max(row[:3]) for row in calculate_online_initialisation.covariances_vector[unknown_anchor.anchor_ID]]), str(calculate_online_initialisation.verifications_vector[unknown_anchor.anchor_ID]), str(calculate_online_initialisation.pos_delta_vector[unknown_anchor.anchor_ID]), str(calculate_online_initialisation.error_vector[unknown_anchor.anchor_ID]), str(calculate_online_initialisation.constant_bias_error_vector[unknown_anchor.anchor_ID]), str(calculate_online_initialisation.linear_bias_error_vector[unknown_anchor.anchor_ID])]

        calculate_online_initialisation.save_row_to_csv(path, row)

def run_simulation_non_linear_least_squares_comparison():
    """Run the simulation to gather measurements and compute a non-linear least squares estimation for the unknown anchor, using different optimisation methods.
    A stopping criterion is used to extract a subset of the measurements, and the results are stored in a CSV file."""

    # Define the path of the csv to store the data
    path = csv_dir / 'non_linear_least_squares.csv'

    # Iterate over environments
    for environment in range(100000):

        calculate_online_initialisation = CalculateOnlineInitialisation()

        # Randomise the environment
        calculate_online_initialisation.randomise_environment()

        unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]
        calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
        calculate_online_initialisation.uwb_online_initialisation.trajectory = calculate_online_initialisation.drone_sim.drone_trajectory
        
        calculate_online_initialisation.run_pipeline_pre_optimisation(None, unknown_anchor.anchor_ID)

        # Use the stopping criterion to extract a subset of the measurements
        try:
            positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]
            distances = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["distances_pre_rough_estimate"]
            measurement_vector = []
            for position, distance in zip(positions, distances):
                measurement_vector.append([position[0], position[1], position[2], distance])
        except:
            continue
        

        length = len(measurement_vector)
        measurement_vector = measurement_vector[:int(length//1)]
        gdop = calculate_online_initialisation.uwb_online_initialisation.compute_GDOP(measurements=measurement_vector, target_coords=unknown_anchor.get_anchor_coordinates())

        if gdop > 3:
            continue
        
        # Run the linear least squares estimation
        calculate_online_initialisation.uwb_online_initialisation.params['use_linear_bias'] = False
        calculate_online_initialisation.uwb_online_initialisation.params['use_constant_bias'] = False
        estimator_linear, covariance_linear, residuals_linear, _ = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(measurement_vector)

        # Run the non-linear least squares estimation with different optimisation methods
        non_linear_methods = ['LM', 'IRLS', 'MM', 'EM']
        estimators = []
        covariances = []
        residualss = []

        for method in non_linear_methods:
            calculate_online_initialisation.uwb_online_initialisation.params['non_linear_optimisation_type'] = method
            estimator, covariance = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_non_linear_least_squares(measurement_vector, estimator_linear)

            estimators.append(estimator)
            covariances.append(covariance)


        # Calculate the error values
        linear_error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimator_linear[:3])
        linear_error_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimator_linear)
        linear_linear_bias_error = np.linalg.norm(estimator_linear[4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_constant_bias_error = np.linalg.norm(estimator_linear[3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)


        error_lm = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimators[0][:3])
        error_lm_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[0])

        error_irls = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimators[1][:3])
        error_irls_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[1])

        error_mm = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimators[2][:3])
        error_mm_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[2])

        error_em = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimators[3][:3])
        error_em_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[3])

        # Calculate metrics to store in the CSV
        number_of_measurements = len(measurement_vector)

        min_angle_between_two_consecutive_measurements = np.rad2deg(np.atan(calculate_online_initialisation.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold']))

        anchor_position_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
        anchor_distances_gt = np.linalg.norm([np.array(row[:3]) - np.array(anchor_position_gt) for row in measurement_vector] , axis=1)

        measured_noise_mean = np.mean(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))
        measured_noise_var = np.var(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))

        min_distance_to_anchor = np.min(anchor_distances_gt)
        max_distance_to_anchor = np.max(anchor_distances_gt)
        mean_distance_to_anchor = np.mean(anchor_distances_gt)
        std_distance_to_anchor = np.std(anchor_distances_gt)

        angular_span_azimuth, angular_span_elevation = calculate_online_initialisation.compute_angular_span_around_point([row[:3] for row in measurement_vector], anchor_position_gt)

        constant_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].bias
        linear_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias

        linear_bias_error_lm = np.linalg.norm(estimators[0][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_bias_error_irls = np.linalg.norm(estimators[1][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_bias_error_krr = np.linalg.norm(estimators[2][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        
        
        constant_bias_error_lm = np.linalg.norm(estimators[0][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)
        constant_bias_error_irls = np.linalg.norm(estimators[1][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)
        constant_bias_error_krr = np.linalg.norm(estimators[2][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)

        noise_variance = calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance
        outlier_probability = calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability

        row = [number_of_measurements, min_angle_between_two_consecutive_measurements, min_distance_to_anchor, max_distance_to_anchor, mean_distance_to_anchor, std_distance_to_anchor, angular_span_elevation, angular_span_azimuth, constant_bias_gt, linear_bias_gt, measured_noise_mean, noise_variance, measured_noise_var, outlier_probability, linear_error, error_lm, error_irls, error_mm, error_em]


        calculate_online_initialisation.save_row_to_csv(path, row)

def run_simulation_trajectory_optimisation_comparison():


    # Define the path of the csv to store the data
    path = csv_dir / 'trajectory_optimisation_comparison.csv'

    # Iterate over environments
    for environment in range(100000):

        calculate_online_initialisation = CalculateOnlineInitialisation()
        calculate_online_initialisation.randomise_environment()

        unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]

        calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
        trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
        trajectory_copy2 = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
        calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy

        setting = "reweighted_linear_both_biases"
        # calculate_online_initialisation.write_settings(setting)

        # calculate_online_initialisation.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted" # Method to use for the rough estimate, either simple_linear or linear_reweighted
        # calculate_online_initialisation.uwb_online_initialisation.params['use_constant_bias'] = True
        # calculate_online_initialisation.uwb_online_initialisation.params['use_linear_bias'] = True
        # calculate_online_initialisation.uwb_online_initialisation.params['zscore_threshold'] = 3
        # calculate_online_initialisation.uwb_online_initialisation.params['regularise'] = True
        calculate_online_initialisation.uwb_online_initialisation.params["trajectory_optimisation_method"] = "GDOP"

        # calculate_online_initialisation.run_pipeline_pre_optimisation()
        calculate_online_initialisation.run_pipeline_pre_optimisation(anchor_id="4")

        weight_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["linear_ls_weights"]
        positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]
        distances = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["distances_pre_rough_estimate"]
        residual_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["residual_vector"]

        anchor_estimate_variance = np.diag(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["covariance_matrix_rough_linear"])[:3]

        if calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["status"] != "optimised_trajectory":
            

            anchor_estimator = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"]
            anchor_estimate = anchor_estimator[:3]
            previous_measurement_positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]

            # To be adjusted if needed
            initial_trajectory = calculate_online_initialisation.uwb_online_initialisation.trajectory
            remaining_trajectory = initial_trajectory
            
            calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.method = calculate_online_initialisation.uwb_online_initialisation.params["trajectory_optimisation_method"]
            # Optimize the trajectory using the previous measurements and the rough estimate of the anchor 
            drone_position = previous_measurement_positions[-1]

            optimal_waypoints = calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.optimize_waypoints_incrementally_spherical(drone_position, anchor_estimator, anchor_estimate_variance, previous_measurement_positions, remaining_trajectory, radius_of_search = 0.4, max_waypoints=10, marginal_gain_threshold=0.00001)
            calculate_online_initialisation.uwb_online_initialisation.current_optimal_waypoints = optimal_waypoints
            calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["optimal_waypoints"] = optimal_waypoints
            # Create a spline trajectory from the optimal waypoints, to collect the measurements but also go back to the initial mission
            # optimal_trajectory = calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.create_optimal_trajectory(previous_measurement_positions[-1], optimal_waypoints)
            # full_end_trajectory = calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.create_full_optimised_trajectory(previous_measurement_positions[-1], optimal_waypoints, initial_trajectory)
            
            # calculate_online_initialisation.uwb_online_initialisation.spline_x_optimal, calculate_online_initialisation.uwb_online_initialisation.spline_y_optimal, calculate_online_initialisation.uwb_online_initialisation.spline_z_optimal = optimal_trajectory.spline_x, optimal_trajectory.spline_y, optimal_trajectory.spline_z
            # calculate_online_initialisation.uwb_online_initialisation.spline_x, calculate_online_initialisation.uwb_online_initialisation.spline_y, calculate_online_initialisation.uwb_online_initialisation.spline_z = full_end_trajectory.spline_x, full_end_trajectory.spline_y, full_end_trajectory.spline_z
            

        calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["status"] = "optimised_trajectory"
        try:
            calculate_online_initialisation.run_pipeline_post_optimisation(None, unknown_anchor.anchor_ID)
        except:
            continue

        # Save the GDOP version

        positions_optimal_GDOP = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_post_rough_estimate"]
        full_estimator_final_GDOP = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator"]

        # CLEANUP
        calculate_online_initialisation.uwb_online_initialisation.reset_measurements_post_rough_initialisation(unknown_anchor.anchor_ID)

        
        calculate_online_initialisation.uwb_online_initialisation.params["trajectory_optimisation_method"] = "FIM"
        anchor_estimator = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"]
        anchor_estimate = anchor_estimator[:3]
        previous_measurement_positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]

        # To be adjusted if needed
        initial_trajectory = calculate_online_initialisation.uwb_online_initialisation.trajectory
        remaining_trajectory = initial_trajectory
        
        calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.method = calculate_online_initialisation.uwb_online_initialisation.params["trajectory_optimisation_method"]
        # Optimize the trajectory using the previous measurements and the rough estimate of the anchor 
        drone_position = previous_measurement_positions[-1]
        optimal_waypoints = calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.optimize_waypoints_incrementally_spherical(drone_position, anchor_estimator, anchor_estimate_variance, previous_measurement_positions, remaining_trajectory, radius_of_search = 0.4, max_waypoints=10, marginal_gain_threshold=0.00001)
        calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["optimal_waypoints"] = optimal_waypoints
        calculate_online_initialisation.uwb_online_initialisation.current_optimal_waypoints = optimal_waypoints
        # Create a spline trajectory from the optimal waypoints, to collect the measurements but also go back to the initial mission
        optimal_waypoints = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["optimal_waypoints"]
        optimal_waypoint_objects = [Waypoint(*waypoint) for waypoint in optimal_waypoints]
        optimal_trajectory_object = Trajectory()
        optimal_trajectory_object.construct_trajectory_linear(optimal_waypoint_objects)
        optimal_trajectory = optimal_trajectory_object.spline_x, optimal_trajectory_object.spline_y, optimal_trajectory_object.spline_z

        calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["status"] == "optimised_trajectory"
        try:
            calculate_online_initialisation.run_pipeline_post_optimisation(None, unknown_anchor.anchor_ID)
        except:
            continue
        positions_optimal_FIM = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_post_rough_estimate"]
        full_estimator_final_FIM = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator"]


        # Save the FIM version


        measurement_vector = []
        for position, distance in zip(positions, distances):
            measurement_vector.append([position[0], position[1], position[2], distance])

        
        anchor_position = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
        anchor_estimator_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator()
        drone_trajectory = calculate_online_initialisation.drone_sim.drone_trajectory
        # optimal_trajectory = calculate_online_initialisation.uwb_online_initialisation.spline_x, calculate_online_initialisation.uwb_online_initialisation.spline_y, calculate_online_initialisation.uwb_online_initialisation.spline_z
        estimated_anchor_position_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][:3]
        estimated_anchor_position_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"][:3]
        

        full_estimator_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"]
        full_estimator_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"]
        full_estimator_final = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator"]
        anchor_estimator_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator()



        error_linear = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_linear)
        print(f"Error_linear: {error_linear:.2f} m")
        error_non_linear = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_non_linear)
        print(f"Error_non_linear: {error_non_linear:.2f} m")
        error_final = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, full_estimator_final[:3])
        error_final_GDOP = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, full_estimator_final_GDOP[:3])
        error_final_FIM = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, full_estimator_final_FIM[:3])
        print(f"Error_final_FIM: {error_final_FIM:.2f} m")
        print(f"Error_final_GDOP: {error_final_GDOP:.2f} m")

        print("Estimator linear", estimated_anchor_position_linear)
        print("Ground truth", anchor_position)

        print("Full estimator linear", full_estimator_linear)
        print("Full estimator non-linear", full_estimator_non_linear)
        print("Full Ground truth", calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator())
        row = [error_linear, error_non_linear, error_final_FIM, error_final_GDOP]

        calculate_online_initialisation.save_row_to_csv(path, row)

def run_simulation_visualisation():

    calculate_online_initialisation = CalculateOnlineInitialisation()
    calculate_online_initialisation.randomise_environment()

    unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]
    calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
    trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
    trajectory_copy2 = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
    calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy

    setting = "linear_reweighted"
    calculate_online_initialisation.write_settings(setting)
    # calculate_online_initialisation.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted" # Method to use for the rough estimate, either simple_linear or linear_reweighted
    # calculate_online_initialisation.uwb_online_initialisation.params['use_constant_bias'] = True
    # calculate_online_initialisation.uwb_online_initialisation.params['use_linear_bias'] = True
    # calculate_online_initialisation.uwb_online_initialisation.params['zscore_threshold'] = 3
    calculate_online_initialisation.uwb_online_initialisation.params['regularise'] = True

    # calculate_online_initialisation.run_pipeline_pre_optimisation()
    calculate_online_initialisation.run_pipeline_full(anchor_id=unknown_anchor.anchor_ID)
    weight_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["linear_ls_weights"]
    positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]
    distances = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["distances_pre_rough_estimate"]
    positions_optimal = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_post_rough_estimate"]
    residual_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["residual_vector"]


    measurement_vector = []
    for position, distance in zip(positions, distances):
        measurement_vector.append([position[0], position[1], position[2], distance])

    
    anchor_position = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
    anchor_estimator_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator()
    drone_trajectory = calculate_online_initialisation.drone_sim.drone_trajectory
    optimal_trajectory = calculate_online_initialisation.uwb_online_initialisation.spline_x, calculate_online_initialisation.uwb_online_initialisation.spline_y, calculate_online_initialisation.uwb_online_initialisation.spline_z
    estimated_anchor_position_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][:3]
    estimated_anchor_position_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"][:3]
    

    full_estimator_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"]
    full_estimator_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"]
    full_estimator_final = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator"]
    anchor_estimator_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator()

    # print(f"Verification vector: {calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]['verification_vector']}")
    # print("\n")
    print(f"Outlier probability: {calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability}")
    print(f"Noise variance: {calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance}")


    error_linear = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_linear)
    print(f"Error_linear: {error_linear:.2f} m")
    error_non_linear = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_non_linear)
    print(f"Error_non_linear: {error_non_linear:.2f} m")
    error_final = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, full_estimator_final[:3])
    print(f"Error_final: {error_final:.2f} m")

    print("Estimator linear", estimated_anchor_position_linear)
    print("Ground truth", anchor_position)

    print("Full estimator linear", full_estimator_linear)
    print("Full estimator non-linear", full_estimator_non_linear)
    print("Full Ground truth", calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator())

    A,b = calculate_online_initialisation.uwb_online_initialisation.setup_linear_least_square(measurement_vector)
    norm_anchor_squared = np.linalg.norm(anchor_position)**2
    x = [anchor_estimator_gt[0], anchor_estimator_gt[1], anchor_estimator_gt[2], 1/anchor_estimator_gt[4]**2, anchor_estimator_gt[3]/anchor_estimator_gt[4]**2, anchor_estimator_gt[3]**2/anchor_estimator_gt[4]**2 - np.linalg.norm(anchor_estimator_gt[:3])**2]
    # x = [anchor_estimator_gt[0]/norm_anchor_squared, anchor_estimator_gt[1]/norm_anchor_squared, anchor_estimator_gt[2]/norm_anchor_squared, 1/anchor_estimator_gt[4]**2/norm_anchor_squared, anchor_estimator_gt[3]/anchor_estimator_gt[4]**2/norm_anchor_squared, 1/norm_anchor_squared]
    ground_truth_residuals =  b - A @ x

    plt.figure(figsize=(10, 10))

    # Create a grid with 2 columns, with the right column split into three rows
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])

    # First subplot (1/3 of the width)
    ax1 = plt.subplot(gs[:, 0], projection='3d')
    ax1.scatter3D([row[0] for row in positions], [row[1] for row in positions], [row[2] for row in positions])
    if len(positions_optimal) > 0:
        ax1.scatter3D([row[0] for row in positions_optimal], [row[1] for row in positions_optimal], [row[2] for row in positions_optimal], c='r')
    ax1.scatter3D(anchor_position[0], anchor_position[1], anchor_position[2], c='r', label='Anchor position')
    ax1.scatter3D(estimated_anchor_position_linear[0], estimated_anchor_position_linear[1], estimated_anchor_position_linear[2], c='k', label='Estimated anchor position Linear')
    ax1.scatter3D(estimated_anchor_position_non_linear[0], estimated_anchor_position_non_linear[1], estimated_anchor_position_non_linear[2], c='b', label='Estimated anchor position Non-Linear')
    ax1.plot3D(trajectory_copy2.spline_x, trajectory_copy2.spline_y, trajectory_copy2.spline_z, c='b', label='Drone trajectory')
    if optimal_trajectory[0] is not None:
        ax1.plot3D(optimal_trajectory[0], optimal_trajectory[1], optimal_trajectory[2], c='g', label='Optimal trajectory')
    ax1.scatter3D(positions[0][0], positions[0][1], positions[0][2], marker='x', c='g', label='Starting position', s=100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # Second subplot (Top right, 2/3 of the width, upper part)
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_ylabel('Weights')
    colors = plt.cm.jet(np.linspace(0, 1, len(weight_vectors)))
    for i, weight_vector in enumerate(weight_vectors):
        ax2.plot(weight_vector, color=colors[i])

    # Create a twin Axes sharing the x-axis
    ax3 = ax2.twinx()
    ax3.set_ylabel('Residuals')
    ax3.scatter(range(len(ground_truth_residuals)), ground_truth_residuals, label='Ground truth residuals', c='b')
    ax3.legend()

    # Center the y-axis around zero for ax2 and ax3
    y_lim_2 = max(abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
    y_lim_3 = max(abs(ax3.get_ylim()[0]), abs(ax3.get_ylim()[1]))

    ax2.set_ylim(-y_lim_2, y_lim_2)
    ax3.set_ylim(-y_lim_3, y_lim_3)

    # Plot vertical lines where residual is bigger than z-score threshold of 2
    zscore_thresh = 2
    zscores = calculate_online_initialisation.uwb_online_initialisation.compute_z_score(ground_truth_residuals)
    outliers = np.where(np.abs(zscores) > zscore_thresh)[0]
    for outlier in outliers:
        ax2.axvline(x=outlier, color='r', linestyle='--')


    # Third subplot (Bottom right, 2/3 of the width, lower part)
    ax4 = plt.subplot(gs[1, 1])
    ax4.set_ylabel('Residuals')
    colors = plt.cm.jet(np.linspace(0, 1, len(residual_vectors)))
    for i, residual_vector in enumerate(residual_vectors):
        ax4.plot(range(len(residual_vector)),residual_vector, color=colors[i])
    for outlier in outliers:
        ax4.axvline(x=outlier, color='r', linestyle='--')

    ax5 = ax4.twinx()
    ax5.set_ylabel('Error')
    ax5.set_yscale('log')
    # Create a scatter plot with a gradient color map
    x_values = np.arange(len(calculate_online_initialisation.error_vector))
    y_values = calculate_online_initialisation.error_vector
    scatter = ax5.scatter(x_values, y_values, c=x_values, cmap='jet', s=50)

    ax6 = plt.subplot(gs[2, 1])
    ax6.set_ylabel('Error')
    #ax6.plot(calculate_online_initialisation.error_vector, label='Error')
    ax6.plot(calculate_online_initialisation.constant_bias_error_vector[unknown_anchor.anchor_ID], label='Constant bias error')
    ax6.plot(calculate_online_initialisation.linear_bias_error_vector[unknown_anchor.anchor_ID], label='Linear bias error')
    ax6.plot(calculate_online_initialisation.constant_bias_vector[unknown_anchor.anchor_ID], label='Constant bias')
    ax6.plot(calculate_online_initialisation.linear_bias_vector[unknown_anchor.anchor_ID], label='Linear bias')

    ax6.legend()
    ax7 = ax6.twinx()
    ax7.plot(calculate_online_initialisation.error_vector[unknown_anchor.anchor_ID], label='Error GT', c='r')
    ax7.set_yscale('log')
    ax7.legend()


    plt.tight_layout()
    plt.show()

def run_simulation_visualisation_trajectory_optimisation():

    calculate_online_initialisation = CalculateOnlineInitialisation()
    calculate_online_initialisation.randomise_environment()

    unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]

    calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
    trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
    trajectory_copy2 = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
    calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy

    setting = "reweighted_linear_both_biases"
    # calculate_online_initialisation.write_settings(setting)

    # calculate_online_initialisation.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted" # Method to use for the rough estimate, either simple_linear or linear_reweighted
    # calculate_online_initialisation.uwb_online_initialisation.params['use_constant_bias'] = True
    # calculate_online_initialisation.uwb_online_initialisation.params['use_linear_bias'] = True
    # calculate_online_initialisation.uwb_online_initialisation.params['zscore_threshold'] = 3
    # calculate_online_initialisation.uwb_online_initialisation.params['regularise'] = True
    calculate_online_initialisation.uwb_online_initialisation.params["trajectory_optimisation_method"] = "GDOP"

    # calculate_online_initialisation.run_pipeline_pre_optimisation()
    calculate_online_initialisation.run_pipeline_pre_optimisation(anchor_id="4")

    weight_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["linear_ls_weights"]
    positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]
    distances = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["distances_pre_rough_estimate"]
    residual_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["residual_vector"]

    anchor_estimate_variance = np.diag(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["covariance_matrix_rough_linear"])[:3]

    if calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["status"] != "optimised_trajectory":
        

        anchor_estimator = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"]
        anchor_estimate = anchor_estimator[:3]
        previous_measurement_positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]

        # To be adjusted if needed
        initial_trajectory = calculate_online_initialisation.uwb_online_initialisation.trajectory
        remaining_trajectory = initial_trajectory
        
        calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.method = calculate_online_initialisation.uwb_online_initialisation.params["trajectory_optimisation_method"]
        # Optimize the trajectory using the previous measurements and the rough estimate of the anchor 
        drone_position = previous_measurement_positions[-1]

        optimal_waypoints = calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.optimize_waypoints_incrementally_spherical(drone_position, anchor_estimator, anchor_estimate_variance, previous_measurement_positions, remaining_trajectory, radius_of_search = 0.4, max_waypoints=10, marginal_gain_threshold=0.00001)
        calculate_online_initialisation.uwb_online_initialisation.current_optimal_waypoints = optimal_waypoints
        calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["optimal_waypoints"] = optimal_waypoints
        # Create a spline trajectory from the optimal waypoints, to collect the measurements but also go back to the initial mission
        # optimal_trajectory = calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.create_optimal_trajectory(previous_measurement_positions[-1], optimal_waypoints)
        # full_end_trajectory = calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.create_full_optimised_trajectory(previous_measurement_positions[-1], optimal_waypoints, initial_trajectory)
        
        # calculate_online_initialisation.uwb_online_initialisation.spline_x_optimal, calculate_online_initialisation.uwb_online_initialisation.spline_y_optimal, calculate_online_initialisation.uwb_online_initialisation.spline_z_optimal = optimal_trajectory.spline_x, optimal_trajectory.spline_y, optimal_trajectory.spline_z
        # calculate_online_initialisation.uwb_online_initialisation.spline_x, calculate_online_initialisation.uwb_online_initialisation.spline_y, calculate_online_initialisation.uwb_online_initialisation.spline_z = full_end_trajectory.spline_x, full_end_trajectory.spline_y, full_end_trajectory.spline_z
        

    calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["status"] = "optimised_trajectory"
    calculate_online_initialisation.run_pipeline_post_optimisation(None, unknown_anchor.anchor_ID)


    # Save the GDOP version

    positions_optimal_GDOP = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_post_rough_estimate"]
    full_estimator_final_GDOP = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator"]

    # CLEANUP
    calculate_online_initialisation.uwb_online_initialisation.reset_measurements_post_rough_initialisation(unknown_anchor.anchor_ID)

    
    calculate_online_initialisation.uwb_online_initialisation.params["trajectory_optimisation_method"] = "FIM"
    anchor_estimator = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"]
    anchor_estimate = anchor_estimator[:3]
    previous_measurement_positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]

    # To be adjusted if needed
    initial_trajectory = calculate_online_initialisation.uwb_online_initialisation.trajectory
    remaining_trajectory = initial_trajectory
    
    calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.method = calculate_online_initialisation.uwb_online_initialisation.params["trajectory_optimisation_method"]
    # Optimize the trajectory using the previous measurements and the rough estimate of the anchor 
    drone_position = previous_measurement_positions[-1]
    optimal_waypoints = calculate_online_initialisation.uwb_online_initialisation.trajectory_optimiser.optimize_waypoints_incrementally_spherical(drone_position, anchor_estimator, anchor_estimate_variance, previous_measurement_positions, remaining_trajectory, radius_of_search = 0.4, max_waypoints=10, marginal_gain_threshold=0.00001)
    calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["optimal_waypoints"] = optimal_waypoints
    calculate_online_initialisation.uwb_online_initialisation.current_optimal_waypoints = optimal_waypoints
    # Create a spline trajectory from the optimal waypoints, to collect the measurements but also go back to the initial mission
    optimal_waypoints = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["optimal_waypoints"]
    optimal_waypoint_objects = [Waypoint(*waypoint) for waypoint in optimal_waypoints]
    optimal_trajectory_object = Trajectory()
    optimal_trajectory_object.construct_trajectory_linear(optimal_waypoint_objects)
    optimal_trajectory = optimal_trajectory_object.spline_x, optimal_trajectory_object.spline_y, optimal_trajectory_object.spline_z

    calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["status"] == "optimised_trajectory"

    calculate_online_initialisation.run_pipeline_post_optimisation(None, unknown_anchor.anchor_ID)

    positions_optimal_FIM = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_post_rough_estimate"]
    full_estimator_final_FIM = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator"]


    # Save the FIM version


    measurement_vector = []
    for position, distance in zip(positions, distances):
        measurement_vector.append([position[0], position[1], position[2], distance])

    
    anchor_position = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
    anchor_estimator_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator()
    drone_trajectory = calculate_online_initialisation.drone_sim.drone_trajectory
    # optimal_trajectory = calculate_online_initialisation.uwb_online_initialisation.spline_x, calculate_online_initialisation.uwb_online_initialisation.spline_y, calculate_online_initialisation.uwb_online_initialisation.spline_z
    estimated_anchor_position_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][:3]
    estimated_anchor_position_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"][:3]
    

    full_estimator_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"]
    full_estimator_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"]
    full_estimator_final = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator"]
    anchor_estimator_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator()

    # print(f"Verification vector: {calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]['verification_vector']}")
    # print("\n")
    print(f"Outlier probability: {calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability}")
    print(f"Noise variance: {calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance}")


    error_linear = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_linear)
    print(f"Error_linear: {error_linear:.2f} m")
    error_non_linear = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_non_linear)
    print(f"Error_non_linear: {error_non_linear:.2f} m")
    error_final = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, full_estimator_final[:3])
    error_final_GDOP = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, full_estimator_final_GDOP[:3])
    error_final_FIM = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, full_estimator_final_FIM[:3])
    print(f"Error_final_FIM: {error_final_FIM:.2f} m")
    print(f"Error_final_GDOP: {error_final_GDOP:.2f} m")

    print("Estimator linear", estimated_anchor_position_linear)
    print("Ground truth", anchor_position)

    print("Full estimator linear", full_estimator_linear)
    print("Full estimator non-linear", full_estimator_non_linear)
    print("Full Ground truth", calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator())

    def calculate_residuals(measurements, gt_position):
        residuals = []
        for measurement in measurements:
            distance = np.linalg.norm(np.array(measurement[:3]) - np.array(gt_position))
            residuals.append(distance - measurement[3])
        return residuals
    
    ground_truth_residuals = calculate_residuals(measurement_vector, anchor_position)
    ground_truth_distances = np.linalg.norm([np.array(row[:3]) - np.array(anchor_position) for row in measurement_vector] , axis=1)
    estimated_distances = np.linalg.norm([np.array(row[:3]) - np.array(full_estimator_non_linear[:3]) for row in measurement_vector] , axis=1)
    plt.figure(figsize=(10, 10))

    # Create a grid with 2 columns, with the right column split into three rows
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1, 1])

    # First subplot (1/3 of the width)
    ax1 = plt.subplot(gs[:, 0], projection='3d')
    ax1.scatter3D([row[0] for row in positions], [row[1] for row in positions], [row[2] for row in positions], label='Measurements on initial trajectory')

    if len(positions_optimal_GDOP) > 0:
        ax1.scatter3D([row[0] for row in positions_optimal_GDOP], [row[1] for row in positions_optimal_GDOP], [row[2] for row in positions_optimal_GDOP], c='r', label='Optimal measurements GDOP')
    if len(positions_optimal_FIM) > 0:
        ax1.scatter3D([row[0] for row in positions_optimal_FIM], [row[1] for row in positions_optimal_FIM], [row[2] for row in positions_optimal_FIM], c='g', label='Optimal measurements FIM')

    ax1.scatter3D(anchor_position[0], anchor_position[1], anchor_position[2], c='r', marker='x', label='Anchor position')
    ax1.scatter3D(estimated_anchor_position_linear[0], estimated_anchor_position_linear[1], estimated_anchor_position_linear[2], marker='x', c='k', label='Estimated anchor position Linear')
    ax1.scatter3D(estimated_anchor_position_non_linear[0], estimated_anchor_position_non_linear[1], estimated_anchor_position_non_linear[2], marker='x', c='b', label='Estimated anchor position Non-Linear')
    ax1.plot3D(trajectory_copy2.spline_x, trajectory_copy2.spline_y, trajectory_copy2.spline_z, c='b', label='Drone trajectory')
    if optimal_trajectory[0] is not None:
        ax1.plot3D(optimal_trajectory[0], optimal_trajectory[1], optimal_trajectory[2], c='g', label='Optimal trajectory')
    ax1.scatter3D(positions[0][0], positions[0][1], positions[0][2], marker='x', c='g', label='Starting position', s=100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # Second subplot (Top right, 2/3 of the width, upper part)
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_ylabel('Weights')
    colors = plt.cm.jet(np.linspace(0, 1, len(weight_vectors)))
    for i, weight_vector in enumerate(weight_vectors):
        ax2.plot(weight_vector, color=colors[i])

    # Create a twin Axes sharing the x-axis
    ax3 = ax2.twinx()
    ax3.set_ylabel('Residuals')
    ax3.scatter(range(len(ground_truth_residuals)), ground_truth_residuals, label='Ground truth residuals', c='b')
    ax3.legend()

    # Center the y-axis around zero for ax2 and ax3
    y_lim_2 = max(abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
    y_lim_3 = max(abs(ax3.get_ylim()[0]), abs(ax3.get_ylim()[1]))

    ax2.set_ylim(-y_lim_2, y_lim_2)
    ax3.set_ylim(-y_lim_3, y_lim_3)

    # Plot vertical lines where residual is bigger than z-score threshold of 2
    zscore_thresh = 2
    zscores = calculate_online_initialisation.uwb_online_initialisation.compute_z_score(ground_truth_residuals)
    outliers = np.where(np.abs(zscores) > zscore_thresh)[0]
    for outlier in outliers:
        ax2.axvline(x=outlier, color='r', linestyle='--')


    # Third subplot (Bottom right, 2/3 of the width, lower part)
    ax4 = plt.subplot(gs[1, 1])
    ax4.set_ylabel('Residuals')
    colors = plt.cm.jet(np.linspace(0, 1, len(residual_vectors)))
    for i, residual_vector in enumerate(residual_vectors):
        ax4.plot(range(len(residual_vector)),residual_vector, color=colors[i])
    for outlier in outliers:
        ax4.axvline(x=outlier, color='r', linestyle='--')

    ax5 = ax4.twinx()
    ax5.set_ylabel('Error')
    ax5.set_yscale('log')
    # Create a scatter plot with a gradient color map
    x_values = np.arange(len(calculate_online_initialisation.error_vector[unknown_anchor.anchor_ID]))
    y_values = calculate_online_initialisation.error_vector[unknown_anchor.anchor_ID]
    scatter = ax5.scatter(x_values, y_values, c=x_values, cmap='jet', s=50)

    ax6 = plt.subplot(gs[2, 1])
    ax6.set_ylabel('Error')
    #ax6.plot(calculate_online_initialisation.error_vector, label='Error')
    line1 = ax6.plot(calculate_online_initialisation.constant_bias_error_vector[unknown_anchor.anchor_ID], label='Constant bias error')[0]
    # ax6.plot(calculate_online_initialisation.linear_bias_error_vector[unknown_anchor.anchor_ID], label='Linear bias error')
    line2 = ax6.plot(calculate_online_initialisation.constant_bias_vector[unknown_anchor.anchor_ID], label='Estimated Constant bias')[0]
    # ax6.plot(calculate_online_initialisation.linear_bias_vector[unknown_anchor.anchor_ID], label='Linear bias')

    ax7 = ax6.twinx()
    line3 = ax7.plot(calculate_online_initialisation.error_vector[unknown_anchor.anchor_ID], label='Ground Truth Error', c='r')[0]
    ax7.set_yscale('log')


    # Combine legends from both axes
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]

    # Add the merged legend to one of the axes (ax6 or ax7)
    ax6.legend(lines, labels)
    ax6.set_xlabel('Measurement number')

    ax8 = plt.subplot(gs[3, 1])
    ax8.plot(range(len(measurement_vector)), ground_truth_distances, c='b', label='Ground truth distances')
    ax8.plot(range(len(measurement_vector)), estimated_distances, c='g', label='Estimated distances')
    ax8.scatter(range(len(measurement_vector)), distances, c='r', label='Measured distances')
    for outlier in outliers:
        ax8.axvline(x=outlier, color='r', linestyle='--')
    ax8.set_ylabel('Distance')
    ax8.set_xlabel('Measurement number')
    ax8.legend()

    print(calculate_online_initialisation.uwb_online_initialisation.params)
    plt.tight_layout()
    plt.show()



# template for the simulations
def run_simulation_template():

    calculate_online_initialisation = CalculateOnlineInitialisation()
    calculate_online_initialisation.randomise_environment()

    unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]
    calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
    trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
    calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy
    
    

    # Calculate the measurement vector to use for the simulation (Needed to compare methods on the same data)
    measurement_vector = None
    # measurement_vector = calculate_online_initialisation.gather_measurements()

    # If needed to compare, start the loop here 

    # Set parameters for the simulation here
    calculate_online_initialisation.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted" 

    # Run the simulation
    calculate_online_initialisation.run_pipeline_full(measurement_vector)

    # Save the error values / other metrics before the next iteration with different parameters

    # Prepare the rows to write to the CSV file
    row = [calculate_online_initialisation.error_vector]
    
    csv_file = csv_dir / 'error_vector.csv'
    calculate_online_initialisation.save_row_to_csv(csv_file, row)


    
def run_simulation_real_data():

    anchor_1 = [0.970928, -0.930796, 2.146243]
    anchor_2 = [0.780176, 0.886912, 2.141455]
    anchor_3 = [3.561750, 3.244373, 1.633419]
    anchor_4 = [5.079113, 0.694456, 0.085207]
    anchor_5 = [8.173811, -0.378892, 0.069715]
    anchor_6 = [-0.386296, 3.504234, 1.197495]

    anchor_list = [anchor_1, anchor_2, anchor_3, anchor_4, anchor_5, anchor_6]

    csv_file = csv_dir / 'drone_halle_bags' / 'trajectory_1_2.csv'

    
    anchor_id_to_display = 1

    # Calculate the measurement vector to use for the simulation (Needed to compare methods on the same data)
    measurement_vector = None
    df = pd.read_csv(csv_file)

    # Group the data by 'anchor_id'
    grouped = df.groupby('anchor_id')

    # Create a dictionary to hold the measurement vectors for each anchor
    measurement_vectors = {}

    # Loop through each anchor_id and extract the measurement vectors
    for anchor_id, group in grouped:
        # Create a list of [x, y, z, range] for each anchor
        vectors = group[['tag_position_x', 'tag_position_y', 'tag_position_z', 'range']].values.tolist()
        measurement_vectors[f'measurement_vector_{anchor_id}'] = vectors

    measurement_vector = measurement_vectors[f'measurement_vector_{anchor_id_to_display}']



    calculate_online_initialisation = CalculateOnlineInitialisation()

    calculate_online_initialisation.drone_sim.unknown_anchors = []
    for i, anchor in enumerate(anchor_list):
        calculate_online_initialisation.drone_sim.unknown_anchors.append(Anchor(str(i+1), *anchor))

    unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[(anchor_id_to_display)-1]

    calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
    flown_trajectory = Trajectory()
    flown_waypoints = []
    for measurement in measurement_vector:
        flown_waypoints.append(Waypoint(*measurement[:3]))

    flown_trajectory.construct_trajectory_linear(flown_waypoints)

    calculate_online_initialisation.drone_sim.drone_trajectory = flown_trajectory
    trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
    calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy

    # setting = "linear_reweighted"
    # calculate_online_initialisation.write_settings(setting)


    calculate_online_initialisation.run_pipeline_pre_optimisation(measurement_vector, str(anchor_id_to_display))
    #calculate_online_initialisation.run_pipeline_full()


    weight_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["linear_ls_weights"]
    positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]
    distances = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["distances_pre_rough_estimate"]
    positions_optimal = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_post_rough_estimate"]
    residual_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["residual_vector"]
    print(f"Length of residual vectors: {len(residual_vectors)}")
    print(f"Length of weight vectors: {len(weight_vectors)}")

    print(f"Length of the last residual vector: {len(residual_vectors[-1])}")
    print(f"Length of the last weight vector: {len(weight_vectors[-1])}")

    measurement_vector = []
    for position, distance in zip(positions, distances):
        measurement_vector.append([position[0], position[1], position[2], distance])

    
    anchor_position = eval(f"anchor_{anchor_id_to_display}")
    anchor_estimator_gt = anchor_position + [0,1]
    drone_trajectory = calculate_online_initialisation.drone_sim.drone_trajectory
    optimal_waypoints = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["optimal_waypoints"]
    optimal_waypoint_objects = [Waypoint(*waypoint) for waypoint in optimal_waypoints]
    optimal_trajectory_object = Trajectory()
    optimal_trajectory_object.construct_trajectory_linear(optimal_waypoint_objects)
    optimal_trajectory = optimal_trajectory_object.spline_x, optimal_trajectory_object.spline_y, optimal_trajectory_object.spline_z
    estimated_anchor_position_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][:3]
    estimated_anchor_position_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"][:3]
    

    full_estimator_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"]
    full_estimator_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"]
    full_estimator_final = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator"]

    # print(f"Verification vector: {calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]['verification_vector']}")
    # print("\n")
    print(f"Outlier probability: {calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability}")
    print(f"Noise variance: {calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance}")


    error_linear = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_linear)
    print(f"Error_linear: {error_linear:.2f} m")
    error_non_linear = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_non_linear)
    print(f"Error_non_linear: {error_non_linear:.2f} m")
    error_final = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, full_estimator_final[:3])
    print(f"Error_final: {error_final:.2f} m")

    print("Estimator linear", estimated_anchor_position_linear)
    print("Ground truth", anchor_position)

    print("Full estimator linear", full_estimator_linear)
    print("Full estimator non-linear", full_estimator_non_linear)
    print("Full Ground truth", calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator())

    def calculate_residuals(measurements, gt_position):
        residuals = []
        for measurement in measurements:
            distance = np.linalg.norm(np.array(measurement[:3]) - np.array(gt_position))
            residuals.append(distance - measurement[3])
        return residuals
    
    def gaussian_mixture_model(measurements, gt_position):
        residuals = calculate_residuals(measurements, gt_position)
        residuals = np.array(residuals).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
        gmm.fit(residuals)
        return gmm

    ground_truth_residuals = calculate_residuals(measurement_vector, anchor_position)
    ground_truth_distances = [np.linalg.norm(np.array(row[:3])-np.array(anchor_position)) for row in measurement_vector]
    estimated_distances = [np.linalg.norm(np.array(row[:3])-np.array(estimated_anchor_position_non_linear)) for row in measurement_vector]

    gmm = gaussian_mixture_model(measurement_vector, anchor_position)
    print("\nFitting GMM using ground truth positions...")
    print(f"Means: {gmm.means_}")
    print(f"Covariances: {gmm.covariances_}")
    print(f"Weights: {gmm.weights_}")


    plt.figure(figsize=(10, 10))

    # Create a grid with 2 columns, with the right column split into three rows
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1, 1])

    # First subplot (1/3 of the width)
    ax1 = plt.subplot(gs[:, 0], projection='3d')
    ax1.scatter3D([row[0] for row in positions], [row[1] for row in positions], [row[2] for row in positions])
    if len(positions_optimal) > 0:
        ax1.scatter3D([row[0] for row in positions_optimal], [row[1] for row in positions_optimal], [row[2] for row in positions_optimal], c='r')
    ax1.scatter3D(anchor_position[0], anchor_position[1], anchor_position[2], c='r', label='Anchor position')
    ax1.scatter3D(estimated_anchor_position_linear[0], estimated_anchor_position_linear[1], estimated_anchor_position_linear[2], c='k', label='Estimated anchor position Linear')
    ax1.scatter3D(estimated_anchor_position_non_linear[0], estimated_anchor_position_non_linear[1], estimated_anchor_position_non_linear[2], c='b', label='Estimated anchor position Non-Linear')
    ax1.plot3D(trajectory_copy.spline_x, trajectory_copy.spline_y, trajectory_copy.spline_z, c='b', label='Drone trajectory')
    if optimal_trajectory[0] is not None:
        ax1.plot3D(optimal_trajectory[0], optimal_trajectory[1], optimal_trajectory[2], c='g', label='Optimal trajectory')
    ax1.scatter3D(positions[0][0], positions[0][1], positions[0][2], marker='x', c='g', label='Starting position', s=100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # Second subplot (Top right, 2/3 of the width, upper part)
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_ylabel('Weights')
    colors = plt.cm.jet(np.linspace(0, 1, len(weight_vectors)))
    for i, weight_vector in enumerate(weight_vectors):
        ax2.plot(weight_vector, color=colors[i])

    # Create a twin Axes sharing the x-axis
    ax3 = ax2.twinx()
    ax3.set_ylabel('Residuals')
    ax3.scatter(range(len(ground_truth_residuals)), ground_truth_residuals, label='Ground truth residuals', c='b')
    ax3.legend()

    # Center the y-axis around zero for ax2 and ax3
    y_lim_2 = max(abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
    y_lim_3 = max(abs(ax3.get_ylim()[0]), abs(ax3.get_ylim()[1]))

    ax2.set_ylim(-y_lim_2, y_lim_2)
    ax3.set_ylim(-y_lim_3, y_lim_3)

    # Plot vertical lines where residual is bigger than z-score threshold of 2
    zscore_thresh = 2
    zscores = calculate_online_initialisation.uwb_online_initialisation.compute_z_score(ground_truth_residuals)
    outliers = np.where(np.abs(zscores) > zscore_thresh)[0]
    for outlier in outliers:
        ax2.axvline(x=outlier, color='r', linestyle='--')


    # Third subplot (Bottom right, 2/3 of the width, lower part)
    ax4 = plt.subplot(gs[1, 1])
    ax4.set_ylabel('Residuals')
    colors = plt.cm.jet(np.linspace(0, 1, len(residual_vectors)))
    for i, residual_vector in enumerate(residual_vectors):
        ax4.plot(range(len(residual_vector)),residual_vector, color=colors[i])
    for outlier in outliers:
        ax4.axvline(x=outlier, color='r', linestyle='--')

    ax5 = ax4.twinx()
    ax5.set_ylabel('Error')
    ax5.set_yscale('log')
    # Create a scatter plot with a gradient color map
    x_values = np.arange(len(calculate_online_initialisation.error_vector[str(anchor_id_to_display)]))
    y_values = calculate_online_initialisation.error_vector[str(anchor_id_to_display)]
    scatter = ax5.scatter(x_values, y_values, c=x_values, cmap='jet', s=50)

    ax6 = plt.subplot(gs[2, 1])
    ax6.set_ylabel('Error')
    #ax6.plot(calculate_online_initialisation.error_vector, label='Error')
    ax6.plot(calculate_online_initialisation.constant_bias_vector[str(anchor_id_to_display)], label='Constant bias')
    ax6.plot(calculate_online_initialisation.linear_bias_vector[str(anchor_id_to_display)], label='Linear bias')

    ax6.legend()
    ax7 = ax6.twinx()
    ax7.plot(calculate_online_initialisation.error_vector[str(anchor_id_to_display)], label='Error GT', c='r')
    ax7.set_yscale('log')
    ax7.legend()

    ax8 = plt.subplot(gs[3, 1])
    ax8.plot(range(len(measurement_vector)), ground_truth_distances, c='b', label='Ground truth distances')
    ax8.plot(range(len(measurement_vector)), estimated_distances, c='g', label='Estimated distances')
    ax8.scatter(range(len(measurement_vector)), distances, c='r', label='Measured distances')
    # for outlier in outliers:
    #     ax8.axvline(x=outlier, color='r', linestyle='--')
    ax8.set_ylabel('Distance')
    ax8.set_xlabel('Measurement number')
    ax8.legend()


    plt.tight_layout()
    plt.show()

def get_real_data_statistics():
    anchor_1 = [0.970928, -0.930796, 2.146243]
    anchor_2 = [0.780176, 0.886912, 2.141455]
    anchor_3 = [3.561750, 3.244373, 1.633419]
    anchor_4 = [5.079113, 0.694456, 0.085207]
    anchor_5 = [8.173811, -0.378892, 0.069715]
    anchor_6 = [-0.386296, 3.504234, 1.197495]

    
    anchor_list = [anchor_1, anchor_2, anchor_3, anchor_4, anchor_5, anchor_6]

    errors_linear = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": []}
    errors_non_linear = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": []}
    
    trajectories = []
    # extract one version of the trajectory
    for trajectory_id in range(1,5):
        run = 1
        csv_file = csv_dir / 'drone_halle_bags' / f'trajectory_{trajectory_id}_{run}.csv'
        df = pd.read_csv(csv_file)
        x,y,z = df['tag_position_x'].values, df['tag_position_y'].values, df['tag_position_z'].values
        trajectories.append([x,y,z])

    # plot the trajectories and the anchor positions
    fig1 = plt.figure(figsize=(10, 10))
    ax = fig1.add_subplot(111, projection='3d')
    for idx, trajectory in enumerate(trajectories):
        ax.plot(trajectory[0], trajectory[1], trajectory[2], label=f'Trajectory {idx + 1}')
    for idx, anchor in enumerate(anchor_list):
        ax.scatter(anchor[0], anchor[1], anchor[2], label=f'Anchor {idx + 1}', s=50 )
    ax.legend()
    #plt.show()

    
    for trajectory in range(1,5):
        for anchor in range(1,7):
            for run in range(1,11):
                anchor_id_to_display = anchor
                trajectory_id = trajectory
                
                csv_file = csv_dir / 'drone_halle_bags' / f'trajectory_{trajectory_id}_{run}.csv'
                
                measurement_vector = None
                df = pd.read_csv(csv_file)

                # Group the data by 'anchor_id'
                grouped = df.groupby('anchor_id')

                # Create a dictionary to hold the measurement vectors for each anchor
                measurement_vectors = {}

                # Loop through each anchor_id and extract the measurement vectors
                for anchor_id, group in grouped:
                    # Create a list of [x, y, z, range] for each anchor
                    vectors = group[['tag_position_x', 'tag_position_y', 'tag_position_z', 'range']].values.tolist()
                    measurement_vectors[f'measurement_vector_{anchor_id}'] = vectors
                print("ANCHOR:", anchor_id_to_display)
                measurement_vector = measurement_vectors[f'measurement_vector_{anchor_id_to_display}']



                calculate_online_initialisation = CalculateOnlineInitialisation()
                
                calculate_online_initialisation.drone_sim.unknown_anchors = []
                for i, anchor_pos in enumerate(anchor_list):
                    calculate_online_initialisation.drone_sim.unknown_anchors.append(Anchor(str(i+1), *anchor_pos))

                unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[(anchor_id_to_display)-1]


                calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
                flown_trajectory = Trajectory()
                flown_waypoints = []
                for measurement in measurement_vector:
                    flown_waypoints.append(Waypoint(*measurement[:3]))

                flown_trajectory.construct_trajectory_linear(flown_waypoints)

                calculate_online_initialisation.drone_sim.drone_trajectory = flown_trajectory
                trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
                calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy

                # setting = "linear_reweighted"
                # calculate_online_initialisation.write_settings(setting)


                calculate_online_initialisation.run_pipeline_pre_optimisation(measurement_vector, str(anchor_id_to_display))


                weight_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["linear_ls_weights"]
                positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_pre_rough_estimate"]
                distances = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["distances_pre_rough_estimate"]
                positions_optimal = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["positions_post_rough_estimate"]
                residual_vectors = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["residual_vector"]

                measurement_vector = []
                for position, distance in zip(positions, distances):
                    measurement_vector.append([position[0], position[1], position[2], distance])

                
                anchor_position = eval(f"anchor_{anchor_id_to_display}")
                anchor_estimator_gt = anchor_position + [0,1]
                drone_trajectory = calculate_online_initialisation.drone_sim.drone_trajectory
                optimal_waypoints = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["optimal_waypoints"]
                optimal_trajectory = [None, None, None]
                estimated_anchor_position_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"][:3]
                estimated_anchor_position_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"][:3]
                

                full_estimator_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_linear"]
                full_estimator_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator_rough_non_linear"]
                full_estimator_final = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[unknown_anchor.anchor_ID]["estimator"]

                errors_linear[str(anchor_id_to_display)].append(calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_linear))
                errors_non_linear[str(anchor_id_to_display)].append(calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_position, estimated_anchor_position_non_linear))

    # Plot boxplots of the errors
    fig, ax = plt.subplots(4, 6, figsize=(15, 10))

    for i in range(1,7):
        for trajectory in range(1,5):
            # Extracting the error data
            linear_errors = errors_linear[str(i)][trajectory-1:trajectory-1+10]
            non_linear_errors = errors_non_linear[str(i)][trajectory-1:trajectory-1+10]
            
            # Plot both boxplots side by side on the same axis
            ax[trajectory-1, i-1].boxplot([linear_errors, non_linear_errors], 
                                        showfliers=False, 
                                        tick_labels=["Linear", "Non-Linear"], 
                                        positions=[1, 2])  # Positions to shift the two boxplots
            
            # Add title and labels
            ax[trajectory-1, i-1].set_title(f"Anchor {i} - Trajectory {trajectory}")
            ax[trajectory-1, i-1].set_ylabel("Error (m)")
    plt.tight_layout()
    plt.show()




def run_simulation_drone_halle():

    calculate_online_initialisation = CalculateOnlineInitialisation()

    calculate_online_initialisation.setup_environment_drone_halle()

    

    calculate_online_initialisation.setup_trajectory_drone_halle()

    trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
    print(f"Length of trajectory: {len(trajectory_copy.spline_x)}")
    
    
    # plt.show()
    trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
    calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy
    
    

    # Calculate the measurement vector to use for the simulation (Needed to compare methods on the same data)
    measurement_vector = None
    # measurement_vector = calculate_online_initialisation.gather_measurements()

    # If needed to compare, start the loop here 

    # Set parameters for the simulation here
    calculate_online_initialisation.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted" 

    for anchor in calculate_online_initialisation.drone_sim.unknown_anchors:
        anchor_id = anchor.anchor_ID
        calculate_online_initialisation.reset_metrics(anchor_id)

    # Run the simulation
    calculate_online_initialisation.run_pipeline_multiple_anchors()


    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor('#545353')
    # Create a grid with 2 columns and 2 rows
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1, 1])

    # First subplot (top left)
    ax1 = plt.subplot(gs[:, 0], projection='3d')
    ax1.set_facecolor('#545353') 
    for anchor in calculate_online_initialisation.drone_sim.unknown_anchors:
        anchor_coordinates = anchor.get_anchor_coordinates()
        
        ax1.text(anchor_coordinates[0], anchor_coordinates[1], anchor_coordinates[2], f'Anchor {anchor.anchor_ID}')

    gt_scatter = ax1.scatter([anchor.get_anchor_coordinates()[0] for anchor in calculate_online_initialisation.drone_sim.unknown_anchors], [anchor.get_anchor_coordinates()[1] for anchor in calculate_online_initialisation.drone_sim.unknown_anchors], [anchor.get_anchor_coordinates()[2] for anchor in calculate_online_initialisation.drone_sim.unknown_anchors], label='Ground Truth', color='black', marker='x')

    ax1.plot3D(trajectory_copy.spline_x, trajectory_copy.spline_y, trajectory_copy.spline_z, linestyle='--', c='b', alpha=1,label='Initial drone trajectory')
    # ax1.plot3D(calculate_online_initialisation.drone_trajectory.spline_x, calculate_online_initialisation.drone_trajectory.spline_y, calculate_online_initialisation.drone_trajectory.spline_z, c='g', label='Final drone trajectory')
    
    for anchor_id, anchor in enumerate(calculate_online_initialisation.drone_sim.unknown_anchors):
        trajectory = calculate_online_initialisation.optimal_trajectories[anchor.anchor_ID]
        if trajectory is not None:
            # ax1.plot3D(trajectory.spline_x, trajectory.spline_y, trajectory.spline_z, label=f'Optimal trajectory {anchor_id+1}')
            pass

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # for anchor_id, anchor in enumerate(calculate_online_initialisation.drone_sim.unknown_anchors):
    #     positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["positions_post_rough_estimate"] + calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["positions_post_rough_estimate"]
    #     x_vals = [x for x, y, z in positions]
    #     y_vals = [y for x, y, z in positions]
    #     z_vals = [z for x, y, z in positions] 

    #     ax1.scatter(x_vals, y_vals, z_vals, label=f'Anchor {anchor_id+1} positions')

    all_measurements = []
    measurement_counts = {}
    for anchor in calculate_online_initialisation.drone_sim.unknown_anchors:
        anchor_id = anchor.anchor_ID
        count = len(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["positions_pre_rough_estimate"]) + len(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["positions_post_rough_estimate"])
        measurement_counts[anchor_id] = count

    # Sort anchors by measurement count (ascending order)
    sorted_anchors = sorted(measurement_counts.keys(), key=lambda anchor_id: measurement_counts[anchor_id])

    # Add measurements from anchors with fewer measurements in order
    for anchor_id in sorted_anchors:
        anchor_measurements = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor_id]["positions_post_rough_estimate"]
        all_measurements += anchor_measurements

    # discard duplicates
    all_measurements = list(dict.fromkeys(all_measurements))

    x_vals = [x for x, y, z in all_measurements]
    y_vals = [y for x, y, z in all_measurements]
    z_vals = [z for x, y, z in all_measurements]

    # Create a scatter plot with a gradient color map
    scatter = ax1.scatter(x_vals, y_vals, z_vals, c=np.arange(len(x_vals)), cmap='jet', label='All positions')
    color_bar = plt.colorbar(scatter, ax=ax1, aspect=5)  # Reduce the height with aspect
    color_bar.set_label('Drone trajectory measurement index')  # Set the title for the color bar

    estimated_anchor_positions_linear = []
    estimated_anchor_positions_non_linear = []
    estimated_anchor_positions_final = []
    for anchor in calculate_online_initialisation.drone_sim.unknown_anchors:
        estimated_anchor_positions_linear.append(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator_rough_linear"])
        estimated_anchor_positions_non_linear.append(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator_rough_non_linear"])
        estimated_anchor_positions_final.append(calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator"])

    linear_scatter = ax1.scatter(
        [position[0] for position in estimated_anchor_positions_linear],
        [position[1] for position in estimated_anchor_positions_linear],
        [position[2] for position in estimated_anchor_positions_linear],
        c='r', label='Estimated anchor position Linear', marker='o'
    )

    non_linear_scatter = ax1.scatter(
        [position[0] for position in estimated_anchor_positions_non_linear],
        [position[1] for position in estimated_anchor_positions_non_linear],
        [position[2] for position in estimated_anchor_positions_non_linear],
        c='g', label='Estimated anchor position Non-Linear', marker='o'
    )

    final_scatter = ax1.scatter(
        [position[0] for position in estimated_anchor_positions_final],
        [position[1] for position in estimated_anchor_positions_final],
        [position[2] for position in estimated_anchor_positions_final],
        c='b', label='Estimated anchor position Final', marker='o'
    )

    # Create a custom legend with only three entries
    ax1.legend([linear_scatter, non_linear_scatter, final_scatter, gt_scatter],
            ['Linear', 'Non-Linear', 'Final', 'Ground Truth'], loc='upper right')
    

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    axes = {}

    for anchor in calculate_online_initialisation.drone_sim.unknown_anchors:
        if calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["status"] != "initialised":
            break
        anchor_gt_position = anchor.get_anchor_coordinates()
        anchor_estimated_position = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator"][:3]

        anchor_id = int(anchor.anchor_ID)
        ranges = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["distances_pre_rough_estimate"]
        drone_positions = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["positions_pre_rough_estimate"]
        ground_truth_ranges = np.linalg.norm([np.array(row[:3]) - np.array(anchor_gt_position) for row in drone_positions], axis=1)
        estimated_ranges = np.linalg.norm([np.array(row[:3]) - np.array(anchor_estimated_position) for row in drone_positions], axis=1)
        axes[anchor_id] = plt.subplot(gs[anchor_id-1,1])
        axes[anchor_id].set_title(f"Anchor {anchor_id}")
        axes[anchor_id].scatter(range(len(ranges)), ranges, label='Measured ranges', c='r')
        axes[anchor_id].plot(range(len(ranges)), ground_truth_ranges, label='Ground truth ranges', c='b')
        axes[anchor_id].plot(range(len(ranges)), estimated_ranges, label='Estimated ranges', c='g')
    
    linear_errors = []
    non_linear_errors = []
    final_errors = []
    for anchor in calculate_online_initialisation.drone_sim.unknown_anchors:
        anchor_gt_position = anchor.get_anchor_coordinates()
        anchor_estimated_position_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator_rough_linear"][:3]
        anchor_estimated_position_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator_rough_non_linear"][:3]
        anchor_estimated_position_final = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator"][:3]
        linear_errors.append(calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_gt_position, anchor_estimated_position_linear))
        non_linear_errors.append(calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_gt_position, anchor_estimated_position_non_linear))
        final_errors.append(calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_gt_position, anchor_estimated_position_final))

    print(f"Errors_linear: {linear_errors}")
    print(f"Errors_non_linear: {non_linear_errors}")
    print(f"Errors_final: {final_errors}")

    def save_optimal_waypoints_to_csv():
        for anchor in calculate_online_initialisation.drone_sim.unknown_anchors:
            optimal_waypoints = calculate_online_initialisation.optimal_waypoints[anchor.anchor_ID]
            csv_file = csv_dir / 'optimal_waypoints' / f'optimal_waypoints_{anchor.anchor_ID}.csv'
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["x", "y", "z"])
                for waypoint in optimal_waypoints:
                    writer.writerow(waypoint)

    save_optimal_waypoints_to_csv()

    plt.tight_layout()
    plt.show()

def monte_carlo_simulation_drone_halle():
            
        for env in range(10000):

            calculate_online_initialisation = CalculateOnlineInitialisation()

            calculate_online_initialisation.setup_environment_drone_halle()

            calculate_online_initialisation.setup_trajectory_drone_halle()

            noise_value = calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance

            trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)

            trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
            calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy
            
            

            # Calculate the measurement vector to use for the simulation (Needed to compare methods on the same data)
            measurement_vector = None
            # measurement_vector = calculate_online_initialisation.gather_measurements()

            # If needed to compare, start the loop here 

            # Set parameters for the simulation here
            calculate_online_initialisation.uwb_online_initialisation.params['rough_estimate_method'] = "linear_reweighted" 

            for anchor in calculate_online_initialisation.drone_sim.unknown_anchors:
                anchor_id = anchor.anchor_ID
                calculate_online_initialisation.reset_metrics(anchor_id)

            # Run the simulation
            calculate_online_initialisation.run_pipeline_multiple_anchors()


            linear_errors = []
            non_linear_errors = []
            final_errors = []
            for anchor in calculate_online_initialisation.drone_sim.unknown_anchors:
                anchor_gt_position = anchor.get_anchor_coordinates()
                anchor_estimated_position_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator_rough_linear"][:3]
                anchor_estimated_position_non_linear = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator_rough_non_linear"][:3]
                anchor_estimated_position_final = calculate_online_initialisation.uwb_online_initialisation.anchor_measurements_dictionary[anchor.anchor_ID]["estimator"][:3]
                linear_errors.append(calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_gt_position, anchor_estimated_position_linear))
                non_linear_errors.append(calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_gt_position, anchor_estimated_position_non_linear))
                final_errors.append(calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(anchor_gt_position, anchor_estimated_position_final))

            row = []
            for i in range(4):
                row.append(calculate_online_initialisation.drone_sim.unknown_anchors[i].noise_variance)
                row.extend([linear_errors[i], non_linear_errors[i], final_errors[i]])
            print(row)
            csv_file = csv_dir / 'monte_carlo_drone_halle.csv'
            calculate_online_initialisation.save_row_to_csv(csv_file, row)


def run_simulation_stopping_criterion_sensitivity():

    
    criterions = ['gdop', 'fim', 'condition_number', 'residuals', 'covariance', 'verification_vector', "convergence"]
    

    def find_error_using_stopping_criterion(stopping_criterion, calculate_online_initialisation, thresholds):
        if stopping_criterion == 'gdop':
            thresh = thresholds[0]

            gdop_vector = calculate_online_initialisation.gdop_vector[str(unknown_anchor.anchor_ID)]
            # find the first index where the gdop is lower than the threshold
            index = next((i for i, x in enumerate(gdop_vector) if x < thresh), None)
            if index is None:
                error = np.inf
            else:
                error = calculate_online_initialisation.error_vector[str(unknown_anchor.anchor_ID)][index]

        elif stopping_criterion == 'fim':
            thresh = thresholds[1]
            fim_vector = calculate_online_initialisation.fim_vector[str(unknown_anchor.anchor_ID)]
            # find the first index where the gdop is lower than the threshold
            index = next((i for i, x in enumerate(fim_vector) if 1/np.linalg.det(x) < thresh), None)
            if index is None:
                error = np.inf
            else:
                error = calculate_online_initialisation.error_vector[str(unknown_anchor.anchor_ID)][index]

        elif stopping_criterion == 'condition_number':
            thresh = thresholds[2]
            condition_number = calculate_online_initialisation.condition_number_vector[str(unknown_anchor.anchor_ID)]
            # find the first index where the gdop is lower than the threshold
            index = next((i for i, x in enumerate(condition_number) if x < thresh), None)
            if index is None:
                error = np.inf
            else:
                error = calculate_online_initialisation.error_vector[str(unknown_anchor.anchor_ID)][index]

        elif stopping_criterion == 'residuals':
            thresh = thresholds[3]
            residuals = calculate_online_initialisation.residuals_vector[str(unknown_anchor.anchor_ID)]
            # find the first index where the gdop is lower than the threshold
            index = next((i for i, x in enumerate(residuals) if x < thresh), None)
            if index is None:
                error = np.inf
            else:
                error = calculate_online_initialisation.error_vector[str(unknown_anchor.anchor_ID)][index]

        elif stopping_criterion == 'covariance':
            thresh = thresholds[4]
            covariance = calculate_online_initialisation.covariances_vector[str(unknown_anchor.anchor_ID)]
            # find the first index where the gdop is lower than the threshold
            index = next((i for i, x in enumerate(covariance) if np.max(x) < thresh), None)
            if index is None:
                error = np.inf
            else:
                error = calculate_online_initialisation.error_vector[str(unknown_anchor.anchor_ID)][index]

        elif stopping_criterion == 'verification_vector':
            thresh = thresholds[5]
            verification_vector = calculate_online_initialisation.verifications_vector[str(unknown_anchor.anchor_ID)]
            # find the first index where the gdop is lower than the threshold
            index = next((i for i, x in enumerate(verification_vector) if x < thresh), None)
            if index is None:
                error = np.inf
            else:
                error = calculate_online_initialisation.error_vector[str(unknown_anchor.anchor_ID)][index]


        elif stopping_criterion == 'convergence':
            delta_positions = calculate_online_initialisation.pos_delta_vector[str(unknown_anchor.anchor_ID)]
            thresh_count, thresh = thresholds[7], thresholds[6]

            # Find the first index where delta_positions has thresh_count consecutive values below thresh
            index = next((i for i in range(len(delta_positions) - thresh_count + 1) 
                        if all(delta_positions[i + j] < thresh for j in range(thresh_count))), None)

            if index is None or index + thresh_count >= len(calculate_online_initialisation.error_vector[str(unknown_anchor.anchor_ID)]):
                error = np.inf  # Set error to infinity if no index is found or out of bounds
            else:
                error = calculate_online_initialisation.error_vector[str(unknown_anchor.anchor_ID)][index + thresh_count]


        return error


    # Iterate over environments
    for environment in range(100000):
        print(f"Environment {environment}")
        calculate_online_initialisation = CalculateOnlineInitialisation()

        calculate_online_initialisation.randomise_environment()

        unknown_anchor = calculate_online_initialisation.drone_sim.unknown_anchors[0]

        calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)

        trajectory_copy = copy.deepcopy(calculate_online_initialisation.drone_sim.drone_trajectory)
        calculate_online_initialisation.uwb_online_initialisation.trajectory = trajectory_copy
        
        

        # Calculate the measurement vector to use for the simulation (Needed to compare methods on the same data)
        # measurement_vector = None
        calculate_online_initialisation.gather_measurements()
        measurement_vector = calculate_online_initialisation.measurement_vector[unknown_anchor.anchor_ID]
        length = len(measurement_vector)
        measurement_vector = measurement_vector[:int(length//1)]


        errors = []
        need_to_exit = False
        measurement_number = []

        # Run the simulation
        try:
            measurement_vector_copy = copy.deepcopy(measurement_vector)
            calculate_online_initialisation.run_pipeline_linear(measurement_vector_copy, unknown_anchor.anchor_ID)
            
        except:
            need_to_exit = True
            break
        
        errors = []

        # Define the lower and upper bounds for the thresholds
        lower_bounds = [0.5, 0.1, 75, 0.5, 0.3, 0.5, 0.1, 3]  # Lower bounds for each threshold
        upper_bounds = [5, 2, 200, 2.5, 2, 3, 0.5, 7]  # Upper bounds for each threshold

        # Initialize a list to store randomized thresholds
        randomized_thresholds = []

        # Randomize the first seven thresholds as floating-point numbers
        for low, high in zip(lower_bounds[:-1], upper_bounds[:-1]):  # Exclude the last threshold for now
            randomized_value = np.random.uniform(low, high)
            randomized_thresholds.append(randomized_value)

        # Randomize the last threshold as an integer
        last_threshold = np.random.randint(lower_bounds[-1], upper_bounds[-1] + 1)  # upper bound is exclusive, so add 1
        randomized_thresholds.append(last_threshold)

        for stopping_criterion in criterions:
            error = find_error_using_stopping_criterion(stopping_criterion, calculate_online_initialisation, randomized_thresholds)
            errors.append(error)

        calculate_online_initialisation.reset_metrics(unknown_anchor.anchor_ID)
            
        if need_to_exit:
            continue
        
        row = randomized_thresholds + errors
        
        csv_file = csv_dir / 'criterion_sensitivity.csv'
        calculate_online_initialisation.save_row_to_csv(csv_file, row)


def load_and_plot_trajectory_of_waypoints():

    csv_file = csv_dir / 'optimal_waypoints' / f'full_waypoints.csv'
    df = pd.read_csv(csv_file)
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    # Create a parameter that changes progressively, e.g., based on the index of the DataFrame
    t = np.linspace(0, 1, len(x))  # This can be adjusted based on how you want to parameterize the progression

    # Create the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)
    # Normalize the 't' values to get colors along a colormap
    norm = plt.Normalize(t.min(), t.max())
    colors = plt.cm.viridis(norm(t))  # Choose a colormap like 'viridis'

    # Plot the points with varying color
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.show()





if __name__ == '__main__':
    
    ### WORKIING

    run_simulation_drone_halle()
    # monte_carlo_simulation_drone_halle()
    run_simulation_real_data()
    get_real_data_statistics()

    ### NOT WORKING
    # run_simulation_trajectory_optimisation_comparison()
    # run_simulation_linear_least_squares_comparison()

    # run_simulation_regularisation()
    # run_simulation_reweighted_ls()
    # run_simulation_stopping_metrics_comparison()
    # run_simulation_stopping_criterion_sensitivity()
    
    # 
    # run_simulation_non_linear_least_squares_comparison()
    # run_simulation_visualisation()


    ### TO BE FIXED SLIGHTLY

    # run_simulation_visualisation_trajectory_optimisation()


    # load_and_plot_trajectory_of_waypoints()
