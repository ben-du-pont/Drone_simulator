import numpy as np

from drone_uwb_simulator.UWB_protocol import Anchor
from drone_uwb_simulator.drone_dynamics import Waypoint, Trajectory

import time


class DroneSimulation():

    def __init__(self, dt = 0.05, drone_speed = 1, number_of_waypoints = 15, bounds = [20, 20, 20]):
        """Initialise the drone simulation class."""

        self.waypoints = None # Stores the main Waypoint objects of the drone trajectory
        self.base_anchors = None # Stores the base Anchor objects, that are known to the drone
        self.unknown_anchors = None # Stores the unknown Anchor objects, that are to be estimated by the drone
        self.drone_trajectory = None # Stores the drone trajectory as a Trajectory object, will be created in initialise_environment()

        self.dt = dt # Time step for the drone simulation
        self.drone_speed = drone_speed # Speed of the drone in m/s

        self.drone_progress = 0 # Stores the current progress of the drone in the trajectory (represents an index in the trajectory spline)

        self.drone_position = None # Stores the current position of the drone as a list [x, y, z]

        self.number_of_waypoints = number_of_waypoints # Number of waypoints in the trajectory

        self.environment_bound_x = bounds[0] # Boundaries of the environment in +/- x direction
        self.environment_bound_y = bounds[1] # Boundaries of the environment in +/- y direction
        self.environment_bound_z = bounds[2] # Boundaries of the environment in z direction

        self.initialise_environment() # Initialise the environment, i.e. waypoints, anchors, and trajectory


    def initialise_environment(self):
        """Initialise the environment, i.e. waypoints, base_anchors, unknown_anchors, and drone_trajectory."""

        # Define the waypoints

        # Option 1 - manually
        self.waypoints = np.array([
            Waypoint(0, 0, 0),
            Waypoint(1, 1, 1),
            Waypoint(2, 2, 2),
            Waypoint(3, 1, 3),
            Waypoint(4, 0, 2),
            Waypoint(6, 6, 3),
            Waypoint(2, 5, 3),
            Waypoint(1, 3, 2),
            Waypoint(0, 2, 1),
            Waypoint(-1, 1, 0),
            Waypoint(-2, 0, 0)
        ])

        # Option 2 - randomly
        self.waypoints = [Waypoint(0,0,0)] + [Waypoint(x, y, z) for (x, y, z) in (np.random.random((self.number_of_waypoints-1,3))- [0.5, 0.5, 0])*[self.environment_bound_x*2, self.environment_bound_y*2, self.environment_bound_z]]

        # Option 3 - two waypoints on opposite edges
        x1 = np.random.choice([-self.environment_bound_x, self.environment_bound_x])
        y1 = np.random.uniform(-self.environment_bound_y, self.environment_bound_y)
        z1 = np.random.uniform(0, self.environment_bound_z)

        # Place the second waypoint on the opposite edge to maximize the distance
        x2 = -x1  # Opposite side in the x direction
        y2 = np.random.choice([-self.environment_bound_y, self.environment_bound_y]) if abs(x1) == self.environment_bound_x else np.random.uniform(-self.environment_bound_y, self.environment_bound_y)
        z2 = self.environment_bound_z - z1  # Opposite side in the z direction

        x3 = np.random.uniform(-self.environment_bound_x/2, self.environment_bound_x/2)
        y3 = np.random.uniform(-self.environment_bound_y/2, self.environment_bound_y/2)
        z3 = np.random.uniform(0, self.environment_bound_z/2)

        x4 = np.random.uniform(-self.environment_bound_x/2, self.environment_bound_x/2)
        y4 = np.random.uniform(-self.environment_bound_y/2, self.environment_bound_y/2)
        z4 = np.random.uniform(0, self.environment_bound_z/2)


        self.waypoints = [
            Waypoint(x1, y1, z1),
            Waypoint(x3, y3, z3),
            Waypoint(x4, y4, z4),
            Waypoint(x2, y2, z2)
        ]


        # Define the base_anchors 
        self.base_anchors = np.array([
            Anchor("0", -1, -1, 0, bias = 5, linear_bias = 1.1, noise_variance = 0.5),
            Anchor("1", 1, 0, 0, bias = 5, linear_bias = 1.1, noise_variance = 0.5),
            Anchor("2", 0, 3, 0, bias = 5, linear_bias = 1.1, noise_variance = 0.5),
            Anchor("3", 0, 6, 0, bias = 5, linear_bias = 1.1, noise_variance = 0.5)
        ])
        
        
        x = np.random.uniform(-self.environment_bound_x, self.environment_bound_x)
        y = np.random.uniform(-self.environment_bound_y, self.environment_bound_y)
        z = np.random.uniform(0, self.environment_bound_z*0)

        # Define the unknown anchors
        self.unknown_anchors = np.array([
            Anchor("4", x, y, z, bias = 0.0951, linear_bias = 1.0049, noise_variance = 0.2, outlier_probability=0.2),
        ])

        # Create trajectory
        self.drone_trajectory = Trajectory(speed=self.drone_speed, dt=self.dt)
        self.drone_trajectory.construct_trajectory_spline(self.waypoints)

        self.drone_position = self.waypoints[0].get_coordinates()

    def update_drone_position_kinematic(self):
        """Update the drone position using the kinematic model of the drone, by strictly following the generated discretised trajectory."""
        
        # Find the index i corresponding to the minimum distance
        i = self.drone_progress

        if i < len(self.drone_trajectory.spline_x):
            new_drone_x, new_drone_y, new_drone_z = self.drone_trajectory.spline_x[i], self.drone_trajectory.spline_y[i], self.drone_trajectory.spline_z[i]
            self.drone_progress += 1
        else:
            new_drone_x, new_drone_y, new_drone_z = self.drone_trajectory.spline_x[0], self.drone_trajectory.spline_y[0], self.drone_trajectory.spline_z[0]
            self.drone_progress = 0

        # Return the updated drone positions using self.drone_trajectory (unchanged)
        self.drone_position = [new_drone_x, new_drone_y, new_drone_z]
        
        return new_drone_x, new_drone_y, new_drone_z




    def get_remaining_waypoints(self, drone_position):
        """Will return the waypoints that have not been visited yet by the drone as a np array of Waypoint objects."""
        
        def get_current_position_index(drone_position):
            """ Find the index of the current drone position in the trajectory. """
            current_position = drone_position
            trajectory_points = self.drone_trajectory

            # Find the closest point in the trajectory to the current position
            distances = [np.linalg.norm(np.array(current_position) - np.array([x,y,z])) for x,y,z in zip(trajectory_points.spline_x, trajectory_points.spline_y, trajectory_points.spline_z)]
            current_position_index = np.argmin(distances)
            
            return current_position_index

        def get_waypoint_indices():
            """ Get the indices of the waypoints in the trajectory. """
            trajectory_points = self.drone_trajectory
            waypoint_coords = [wp.get_coordinates() for wp in self.waypoints]

            waypoint_indices = []
            for waypoint in waypoint_coords:
                # Find the index of the waypoint in the trajectory
                distances = [np.linalg.norm(np.array(waypoint) - np.array([x,y,z])) for (x,y,z) in zip(trajectory_points.spline_x, trajectory_points.spline_y, trajectory_points.spline_z)]
                waypoint_index = np.argmin(distances)
                waypoint_indices.append(waypoint_index)
            return waypoint_indices

        def get_unvisited_waypoints(drone_position):
            """ Return a list of waypoints that have not been visited yet. """
            current_position_index = get_current_position_index(drone_position)
            waypoint_indices = get_waypoint_indices()

            unvisited_waypoints = [self.waypoints[i] for i in range(len(self.waypoints)) if waypoint_indices[i] > current_position_index]
            return np.array(unvisited_waypoints)
        
        return get_unvisited_waypoints(drone_position)