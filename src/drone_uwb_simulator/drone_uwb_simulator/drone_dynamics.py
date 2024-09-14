import numpy as np

import scipy.integrate
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid as cumtrapz


# Kinematics 

# Waypoint class to store the coordinates of points through which the drone will pass
class Waypoint:
    """ Represents a waypoint in 3D space."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_coordinates(self):
        """ Return the coordinates as an array. """
        return [self.x, self.y, self.z]
    

    
    
# Trajectory class to construct the trajectory based on a series of waypoints
class Trajectory:
    """ Represents a trajectory through a series of waypoint objects in 3D space.
    The trajectory can be constructed using either a spline or linear interpolation between waypoints."""

    def __init__(self, speed = 3, dt = 0.05):
        """Specify the desired speed and time interval for the trajectory helps to sample the trajectory at a constant speed. """
        self.speed = speed  # Desired speed of the drone (units per second)
        self.dt = dt        # Time interval at which to sample the trajectory

        self.spline_x = np.array([])
        self.spline_y = np.array([])
        self.spline_z = np.array([])

        self.waypoints = []
        

    def calculate_arc_length(self, x, y, z):
        """ Calculates the cumulative arc length of the trajectory. """
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        return np.hstack([0, cumtrapz(distances)])

    def reparameterize_spline(self, arc_length, x, y, z):
        """ Creates new splines based on arc length parametrization. """
        cs_x = CubicSpline(arc_length, x)
        cs_y = CubicSpline(arc_length, y)
        cs_z = CubicSpline(arc_length, z)
        return cs_x, cs_y, cs_z

    def sample_spline(self, cs_x, cs_y, cs_z, total_length):
        """ Samples the spline evenly according to the desired speed and time interval. """
        self.num_points = int(total_length / (self.speed * self.dt))
        even_arc_lengths = np.linspace(0, total_length, self.num_points)
        x = cs_x(even_arc_lengths)
        y = cs_y(even_arc_lengths)
        z = cs_z(even_arc_lengths)
        return x, y, z

    def construct_trajectory_spline(self, waypoints):
        """ Fits a spline through provided waypoints and samples it for constant speed movement. """
        self.waypoints = waypoints
        x, y, z = zip(*[(wp.x, wp.y, wp.z) for wp in waypoints])
        arc_length = self.calculate_arc_length(x, y, z)
        cs_x, cs_y, cs_z = self.reparameterize_spline(arc_length, x, y, z)
        total_length = arc_length[-1]
        self.spline_x, self.spline_y, self.spline_z = self.sample_spline(cs_x, cs_y, cs_z, total_length)

    def find_closest_waypoint(self, current_position):
        """ Finds the index of the closest waypoint to the current position. """
        distances = np.sqrt((self.spline_x - current_position[0])**2 + 
                            (self.spline_y - current_position[1])**2 + 
                            (self.spline_z - current_position[2])**2)
        return np.argmin(distances)
    
    def get_waypoint(self, index):
        """ Returns the waypoint at the specified index. """
        return Waypoint(self.spline_x[index], self.spline_y[index], self.spline_z[index])
    

    def get_lookahead_distance_waypoint(self, current_position, lookahead_distance):
        """ Finds the waypoint at the specified lookahead distance from the current position. """
        closest_index = self.find_closest_waypoint(current_position)
        total_distance = 0
        for i in range(closest_index, len(self.spline_x) - 1):
            total_distance += np.sqrt((self.spline_x[i + 1] - self.spline_x[i])**2 + 
                                      (self.spline_y[i + 1] - self.spline_y[i])**2 + 
                                      (self.spline_z[i + 1] - self.spline_z[i])**2)
            if total_distance >= lookahead_distance:
                return self.get_waypoint(i+1)
        return self.get_waypoint(-1)
    
    def construct_trajectory_linear(self, waypoints):
        """ Creates a linear trajectory between provided waypoints. """

        self.waypoints = waypoints
        if not waypoints:
            # If no waypoints provided, return empty trajectory
            self.spline_x = np.array([])
            self.spline_y = np.array([])
            self.spline_z = np.array([])
            return

        self.spline_x = []
        self.spline_y = []
        self.spline_z = []

        for i in range(len(waypoints) - 1):
            wp_start = waypoints[i]
            wp_end = waypoints[i + 1]

            # Calculate the distance between waypoints
            distance = np.linalg.norm(np.array([wp_end.x - wp_start.x, wp_end.y - wp_start.y, wp_end.z - wp_start.z]))

            # Calculate the number of points between waypoints based on speed and dt
            num_points_between = np.maximum(int(distance / (self.speed * self.dt)), 1)

            # Linearly interpolate between waypoints, excluding start point of first segment
            x_interp = np.linspace(wp_start.x if i == 0 else wp_start.x + (wp_end.x - wp_start.x) / num_points_between,
                                    wp_end.x, num_points_between)
            y_interp = np.linspace(wp_start.y if i == 0 else wp_start.y + (wp_end.y - wp_start.y) / num_points_between,
                                    wp_end.y, num_points_between)
            z_interp = np.linspace(wp_start.z if i == 0 else wp_start.z + (wp_end.z - wp_start.z) / num_points_between,
                                    wp_end.z, num_points_between)

            # Extend the spline arrays with interpolated points
            self.spline_x.extend(x_interp)
            self.spline_y.extend(y_interp)
            self.spline_z.extend(z_interp)

        # Convert spline arrays to numpy arrays
        self.spline_x = np.array(self.spline_x)
        self.spline_y = np.array(self.spline_y)
        self.spline_z = np.array(self.spline_z)