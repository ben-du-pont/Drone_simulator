import numpy as np

from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid as cumtrapz


# Kinematics 

# Waypoint class to store the coordinates of points through which the drone will pass
class Waypoint:
    """ Represents a waypoint for the drone to fly through in 3D space."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_coordinates(self):
        """ Return the coordinates as an array. """
        return [self.x, self.y, self.z]

    
# Trajectory class to construct an array of positions based on a series of waypoints through which the drone will pass
class Trajectory:
    """
    Represents a trajectory through a series of waypoint objects in 3D space.
    The trajectory is constructed using cubic spline interpolation between waypoints,
    ensuring constant speed movement along the path.

    Attributes:
    -----------
    speed : float
        Desired speed of the drone in units per second. Default is 3.
    dt : float
        Time interval at which to sample the trajectory in seconds. Default is 0.05.
    spline_x : np.ndarray
        Array to store x-coordinates of the spline.
    spline_y : np.ndarray
        Array to store y-coordinates of the spline.
    spline_z : np.ndarray
        Array to store z-coordinates of the spline.
    waypoints : list
        List to store waypoints (as waypoint objects) for the drone's trajectory.
    num_points : int
        Number of points in the sampled trajectory.

    Methods:
    --------
    construct_trajectory_linear(waypoints)
        Construct a linear trajectory through provided waypoints.
    calculate_arc_length(cs_x, cs_y, cs_z, t)
        Calculate the actual arc length using spline derivatives.
    construct_trajectory_spline(waypoints)
        Construct a spline trajectory through provided waypoints with constant speed.
    verify_constant_speed()
        Verify that points along the trajectory are approximately equidistant.
    find_closest_waypoint(current_position)
        Find the index of the closest waypoint to the current position.
    get_waypoint(index)
        Get the waypoint at the specified index.
    get_lookahead_distance_waypoint(current_position, lookahead_distance)
        Find the waypoint at the specified lookahead distance from the current position.
    """

    def __init__(self, speed=3, dt=0.05):
        """
        Initialize the drone trajectory with a specified speed and time discretization interval.

        Parameters:
        -----------
        speed : float
            Desired speed of the drone in units per second. Default is 3.
        dt : float
            Time interval at which to sample the trajectory in seconds. Default is 0.05.
        """
        self.speed = speed
        self.dt = dt
        self.spline_x = np.array([])
        self.spline_y = np.array([])
        self.spline_z = np.array([])
        self.waypoints = []
        self.num_points = 0

    def construct_trajectory_linear(self, waypoints):
        """
        Creates a linear trajectory between provided waypoints with constant speed movement.
        
        This method performs the following steps:
        1. Calculates the total path length through all waypoints
        2. Determines the required number of points based on desired speed and time interval
        3. Distributes points along each segment proportionally to segment length
        4. Performs linear interpolation between waypoints
        
        Parameters:
        -----------
        waypoints : list
            A list of waypoint objects, where each waypoint has attributes `x`, `y`, 
            and `z` representing its coordinates.
        
        Returns:
        --------
        None
            Updates the instance variables `spline_x`, `spline_y`, and `spline_z` 
            with the interpolated trajectory points.
        
        Notes:
        ------
        - The method ensures constant speed movement by maintaining equal distances 
        between consecutive points.
        - The number of points in each segment is proportional to the segment length.
        - All generated points lie exactly on the straight lines between waypoints.
        """
        self.waypoints = waypoints
        if not waypoints:
            self.spline_x = np.array([])
            self.spline_y = np.array([])
            self.spline_z = np.array([])
            return

        # Calculate segment lengths and total path length
        segments = []
        total_length = 0
        for i in range(len(waypoints) - 1):
            wp_start = waypoints[i]
            wp_end = waypoints[i + 1]
            
            # Calculate segment vector and length
            segment_vector = np.array([wp_end.x - wp_start.x, 
                                    wp_end.y - wp_start.y, 
                                    wp_end.z - wp_start.z])
            segment_length = np.linalg.norm(segment_vector)
            
            segments.append({
                'start': wp_start,
                'end': wp_end,
                'length': segment_length,
                'vector': segment_vector
            })
            total_length += segment_length

        # Calculate total number of points needed for desired speed
        total_points_needed = max(int(total_length / (self.speed * self.dt)), 1)
        
        # Initialize output arrays
        self.spline_x = np.zeros(total_points_needed)
        self.spline_y = np.zeros(total_points_needed)
        self.spline_z = np.zeros(total_points_needed)
        
        # Distribute points across segments
        current_point = 0
        for segment in segments:
            # Calculate number of points for this segment proportional to its length
            segment_points = int(round((segment['length'] / total_length) * 
                                    (total_points_needed - 1)))
            if segment == segments[-1]:  # Last segment
                segment_points = total_points_needed - current_point
            
            if segment_points > 0:
                # Generate interpolation parameters
                t = np.linspace(0, 1, segment_points)
                
                # Interpolate points
                segment_x = segment['start'].x + t * segment['vector'][0]
                segment_y = segment['start'].y + t * segment['vector'][1]
                segment_z = segment['start'].z + t * segment['vector'][2]
                
                # Store points
                self.spline_x[current_point:current_point + segment_points] = segment_x
                self.spline_y[current_point:current_point + segment_points] = segment_y
                self.spline_z[current_point:current_point + segment_points] = segment_z
                
                current_point += segment_points
        
        # Ensure last point matches final waypoint
        if waypoints:
            self.spline_x[-1] = waypoints[-1].x
            self.spline_y[-1] = waypoints[-1].y
            self.spline_z[-1] = waypoints[-1].z

    def calculate_arc_length(self, cs_x, cs_y, cs_z, t):
        """
        Calculates the actual arc length of the spline using analytical derivatives.
        
        This method computes the arc length by:
        1. Computing the derivatives of the spline in each dimension
        2. Calculating the speed at each point using these derivatives
        3. Integrating the speed to get the arc length

        Parameters:
        -----------
        cs_x : scipy.interpolate.CubicSpline
            Cubic spline for x-coordinate.
        cs_y : scipy.interpolate.CubicSpline
            Cubic spline for y-coordinate.
        cs_z : scipy.interpolate.CubicSpline
            Cubic spline for z-coordinate.
        t : np.ndarray
            Parameter values at which to evaluate the arc length.

        Returns:
        --------
        np.ndarray
            Cumulative arc length at each parameter value.
        """
        # Get derivatives of the splines
        dx_dt = cs_x.derivative()(t)
        dy_dt = cs_y.derivative()(t)
        dz_dt = cs_z.derivative()(t)
        
        # Calculate speed at each point
        speed = np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)
        
        # Integrate speed to get arc length
        return cumtrapz(speed, t, initial=0)

    def construct_trajectory_spline(self, waypoints):
        """
        Constructs a spline trajectory through provided waypoints ensuring constant speed movement.
        
        This method performs the following steps:
        1. Creates initial cubic splines through the waypoints
        2. Calculates the total arc length of the path
        3. Reparameterizes the spline to achieve constant speed movement
        4. Samples the reparameterized spline at equal time intervals

        Parameters:
        -----------
        waypoints : list
            List of Waypoint objects defining the path. Each waypoint should have
            x, y, and z attributes representing its coordinates.

        Returns:
        --------
        None
            Updates the instance variables spline_x, spline_y, and spline_z with
            the interpolated trajectory points.
        """
        self.waypoints = waypoints
        if not waypoints:
            self.spline_x = self.spline_y = self.spline_z = np.array([])
            return

        # Extract coordinates
        x, y, z = zip(*[(wp.x, wp.y, wp.z) for wp in waypoints])
        
        # Initial parameter space
        t = np.linspace(0, 1, len(waypoints))
        
        # Create initial splines
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_z = CubicSpline(t, z)
        
        # Calculate total arc length using fine sampling
        t_fine = np.linspace(0, 1, 1000)
        arc_length = self.calculate_arc_length(cs_x, cs_y, cs_z, t_fine)
        total_length = arc_length[-1]
        
        # Calculate number of points needed for desired speed
        self.num_points = max(int(total_length / (self.speed * self.dt)), 1)
        
        # Create new parameter values that give equal arc length segments
        desired_distances = np.linspace(0, total_length, self.num_points)
        new_t = np.interp(desired_distances, arc_length, t_fine)
        
        # Sample the splines at the new parameter values
        self.spline_x = cs_x(new_t)
        self.spline_y = cs_y(new_t)
        self.spline_z = cs_z(new_t)

    def verify_constant_speed(self):
        """
        Verifies that points along the trajectory are approximately equidistant.
        
        This method checks if the distance between consecutive points matches
        the expected distance (speed * dt) within a reasonable tolerance.

        Returns:
        --------
        tuple
            A tuple containing:
            - max_deviation (float): Maximum deviation from the expected distance
              between consecutive points
            - expected_distance (float): The expected distance between points
              based on speed and dt

        Notes:
        ------
        This method is useful for debugging and verifying that the trajectory
        generation maintains constant speed movement as intended.
        """
        dx = np.diff(self.spline_x)
        dy = np.diff(self.spline_y)
        dz = np.diff(self.spline_z)
        
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        expected_distance = self.speed * self.dt
        
        max_deviation = np.max(np.abs(distances - expected_distance))
        return max_deviation, expected_distance

    def find_closest_waypoint(self, current_position):
        """
        Finds the index of the closest waypoint to the current position.

        Parameters:
        -----------
        current_position : array-like
            A 3-element array-like structure representing the current position 
            in the format [x, y, z].

        Returns:
        --------
        int
            The index of the closest waypoint to the current position.
        """

        distances = np.sqrt((self.spline_x - current_position[0])**2 + 
                            (self.spline_y - current_position[1])**2 + 
                            (self.spline_z - current_position[2])**2)
        
        return np.argmin(distances)
    
    def get_waypoint(self, index):
        """
        Returns the waypoint at the specified index.

        Parameters:
        index (int): The index of the waypoint to retrieve.

        Returns:
        Waypoint: An instance of the Waypoint class representing the waypoint at the specified index.
        """

        return Waypoint(self.spline_x[index], self.spline_y[index], self.spline_z[index])
    
    def get_lookahead_distance_waypoint(self, current_position, lookahead_distance):
        """
        Finds the waypoint at the specified lookahead distance from the current position.

        Parameters:
        current_position (tuple): A tuple (x, y, z) representing the current position of the drone.
        lookahead_distance (float): The distance to look ahead along the path to find the waypoint.

        Returns:
        tuple: A tuple (x, y, z) representing the waypoint at the specified lookahead distance.
               If the lookahead distance exceeds the path length, the last waypoint is returned.
        """
        
        closest_index = self.find_closest_waypoint(current_position)
        total_distance = 0

        for i in range(closest_index, len(self.spline_x) - 1):
            total_distance += np.sqrt((self.spline_x[i + 1] - self.spline_x[i])**2 + 
                                      (self.spline_y[i + 1] - self.spline_y[i])**2 + 
                                      (self.spline_z[i + 1] - self.spline_z[i])**2)
            
            if total_distance >= lookahead_distance:
                return self.get_waypoint(i+1)
            
        return self.get_waypoint(-1)


# Dynamics

#TODO: Add some dynamics to simulate the drone flight based on the trajectory required or the wayponits provided