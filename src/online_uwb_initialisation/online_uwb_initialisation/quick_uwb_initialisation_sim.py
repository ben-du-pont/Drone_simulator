from drone_uwb_simulator.UWB_protocol import Anchor
from drone_uwb_simulator.drone_simulator import DroneSimulation
from drone_uwb_simulator.drone_dynamics import Waypoint, Trajectory

from online_uwb_initialisation.uwb_online_initialisation import UwbOnlineInitialisation
from online_uwb_initialisation.trajectory_optimisation import NonLinearTrajectoryOptimization

import numpy as np
import csv

import random

from pathlib import Path

package_path = Path(__file__).parent.resolve()
csv_dir = package_path

class CalculateOnlineInitialisation:

    def __init__(self):
        self.drone_sim = DroneSimulation()
        self.uwb_online_initialisation = UwbOnlineInitialisation()
        self.optimiser = NonLinearTrajectoryOptimization()



        self.drone_position_post_rough_estimate = None

        self.anchor_rough_estimate_linear = [None, None, None, None, None]
        self.anchor_rough_estimate_non_linear = [None, None, None, None, None]
        self.original_anchor_estimator = [None, None, None, None, None]
        self.final_anchor_estimator  = [None, None, None, None, None]

        self.unknown_anchor = self.drone_sim.unknown_anchors[0]
        self.uwb_online_initialisation.unknown_anchors = self.drone_sim.unknown_anchors

    def get_unkown_anchor_rough_estimate(self):
        self.drone_position_post_rough_estimate = self.uwb_online_initialisation.compute_anchor_rough_estimate(self.drone_sim.drone_trajectory, self.unknown_anchor)

        if self.drone_position_post_rough_estimate is not None:
            self.anchor_rough_estimate_linear = self.uwb_online_initialisation.unknown_anchor_measurements[self.unknown_anchor.anchor_ID]["estimator_rough_linear"]
            self.anchor_rough_estimate_non_linear = self.uwb_online_initialisation.unknown_anchor_measurements[self.unknown_anchor.anchor_ID]["estimator_rough_non_linear"]

    def optimise_trajectory(self):
        if self.anchor_rough_estimate_non_linear is not None:
            self.optimiser.update_anchor_positions(self.anchor_rough_estimate_non_linear[:3])

            self.remaining_waypoints = self.drone_sim.get_remaining_waypoints(self.drone_position_post_rough_estimate)
            self.remaining_waypoint_coordinates = [self.remaining_waypoints[i].get_coordinates() for i in range(len(self.remaining_waypoints))]
            
            self.optimiser.update_waypoints(self.remaining_waypoint_coordinates)

            self.optimised_trajectory_waypoints = self.optimiser.run_optimization()

    def get_final_anchor_estimate(self):
        self.uwb_online_initialisation.refine_anchor_positions(self.optimised_trajectory_waypoints, self.unknown_anchor)
        self.final_anchor_estimator = self.uwb_online_initialisation.unknown_anchor_measurements[self.unknown_anchor.anchor_ID]["estimator"]


    def get_original_anchor_estimate(self):
        self.uwb_online_initialisation.refine_anchor_positions(self.remaining_waypoint_coordinates, self.unknown_anchor)
        self.original_anchor_estimator = self.uwb_online_initialisation.unknown_anchor_measurements[self.unknown_anchor.anchor_ID]["estimator"]


    def full_anchor_initialisation(self):
        try:
            self.get_unkown_anchor_rough_estimate()
            self.optimise_trajectory()
            self.get_final_anchor_estimate()
            self.uwb_online_initialisation.reset_measurements_post_rough_initialisation(self.unknown_anchor.anchor_ID)
            self.get_original_anchor_estimate()
        except:
            self.set_estimators_to_NaN()


    def reset_estimators(self):

        self.drone_position_post_rough_estimate = None

        self.anchor_rough_estimate_linear = [None, None, None, None, None]
        self.anchor_rough_estimate_non_linear = [None, None, None, None, None]
        self.original_anchor_estimator = [None, None, None, None, None]
        self.final_anchor_estimator  = [None, None, None, None, None]

    def set_estimators_to_NaN(self):

        self.drone_position_post_rough_estimate = None

        self.anchor_rough_estimate_linear = [np.nan, np.nan, np.nan, np.nan, np.nan]
        self.anchor_rough_estimate_non_linear = [np.nan, np.nan, np.nan, np.nan, np.nan]
        self.original_anchor_estimator = [np.nan, np.nan, np.nan, np.nan, np.nan]
        self.final_anchor_estimator  = [np.nan, np.nan, np.nan, np.nan, np.nan]


def main():
    # hyperparameters
    distance_to_anchor_threshold_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    gdop_threshold_values = [0.2, 0.5, 0.8, 1, 2, 3, 4, 5]
    weight_angle_variance_values = [0.01, 0.1, 1, 10, 100, 1000]
    weight_distance_variance_values = [0.01, 0.1, 1, 10, 100, 1000]
    weight_deviation_values = [0.01, 0.1, 1, 10, 100, 1000]
    number_of_measurements_values = [1, 2, 3, 4, 5]


    
    
    for environment in range(100000):
        
        
        calculate_online_initialisation = CalculateOnlineInitialisation()

        for i in range(200):
            # Reset all measurements and estimators
            calculate_online_initialisation.reset_estimators()
            calculate_online_initialisation.uwb_online_initialisation.reset_all_measurements(calculate_online_initialisation.unknown_anchor.anchor_ID)

            # Randomly select hyperparameters

            distance_to_anchor_threshold = random.choice(distance_to_anchor_threshold_values)
            gdop_threshold = random.choice(gdop_threshold_values)
            weight_angle = random.choice(weight_angle_variance_values)
            weight_distance = random.choice(weight_distance_variance_values)
            weight_dev = random.choice(weight_deviation_values)
            num_measurements = random.choice(number_of_measurements_values)

            # Set hyperparameters

            calculate_online_initialisation.uwb_online_initialisation.distance_to_anchor_ratio_threshold = distance_to_anchor_threshold
            calculate_online_initialisation.uwb_online_initialisation.gdop_threshold = gdop_threshold
            calculate_online_initialisation.optimiser.update_weights([weight_angle,weight_distance,weight_dev])
            calculate_online_initialisation.uwb_online_initialisation.number_of_measurements = num_measurements

            # Do the full anchor initialisation
            calculate_online_initialisation.full_anchor_initialisation()
            
            
            

            # Save error values to CSV
            with open(csv_dir / 'error_values.csv', mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([
                    environment,
                    i,
                    distance_to_anchor_threshold,
                    gdop_threshold,
                    weight_angle,
                    weight_distance,
                    weight_dev,
                    num_measurements,

                    calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(0, calculate_online_initialisation.anchor_rough_estimate_linear[:3]),
                    calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(0, calculate_online_initialisation.anchor_rough_estimate_linear),

                    calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(0, calculate_online_initialisation.anchor_rough_estimate_non_linear[:3]),
                    calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(0, calculate_online_initialisation.anchor_rough_estimate_non_linear),

                    calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(0, calculate_online_initialisation.original_anchor_estimator[:3]),
                    calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(0, calculate_online_initialisation.original_anchor_estimator),

                    calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(0, calculate_online_initialisation.final_anchor_estimator[:3]),
                    calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(0, calculate_online_initialisation.final_anchor_estimator)
                ])
    
    




if __name__ == '__main__':
    main()