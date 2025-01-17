import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from enum import Enum

from drone_uwb_simulator.UWB_protocol import Anchor, BiasModel, NoiseModel, UWBNetwork
from drone_uwb_simulator.drone_dynamics import Waypoint, Trajectory




@dataclass
class SimulationConfig:
    """
    Configuration parameters for the drone simulation environment.
    
    Attributes:
    -----------
    dt : float
        Time step for simulation in seconds.
    drone_speed : float
        Speed of the drone in meters per second.
    num_waypoints : int
        Number of waypoints to generate for random trajectories.
    bounds : Tuple[float, float, float]
        Environment boundaries as (x_bound, y_bound, z_bound) in meters.
    """
    dt: float = 0.05
    drone_speed: float = 1.0
    num_waypoints: int = 15
    bounds: Tuple[float, float, float] = (5.0, 5.0, 5.0)

class WaypointGenerationMode(Enum):
    """
    Enumeration of available waypoint generation modes.
    
    Modes:
    ------
    MANUAL : Use manually specified waypoints
    RANDOM : Generate random waypoints within bounds
    OPPOSITE_EDGES : Generate waypoints on opposite edges of the environment
    """
    MANUAL = "manual"
    RANDOM = "random"
    OPPOSITE_EDGES = "opposite_edges"

class DroneSimulation:
    """
    Simulates a drone moving through an environment with UWB anchors.
    
    This class manages the complete simulation environment including:
    - Waypoint generation and trajectory planning
    - Base and unknown UWB anchor placement
    - Drone movement and position updates
    
    Attributes:
    -----------
    waypoints : List[Waypoint]
        List of waypoints defining the drone's path.
    base_anchors : List[Anchor]
        Known UWB anchors in the environment.
    unknown_anchors : List[Anchor]
        UWB anchors to be estimated/discovered.
    drone_trajectory : Trajectory
        Computed trajectory through waypoints.
    drone_position : np.ndarray
        Current 3D position of the drone.
    config : SimulationConfig
        Configuration parameters for the simulation.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the drone simulation environment.
        
        Parameters:
        -----------
        config : SimulationConfig, optional
            Configuration parameters for the simulation.
            If None, default configuration is used.
        """
        self.config = config or SimulationConfig()
        
        # Initialize simulation components
        self.waypoints = []
        self.base_anchors = []
        self.unknown_anchors = []
        self.drone_trajectory = None
        
        # Initialize drone state
        self.drone_progress = 0
        self.drone_position = np.zeros(3)
        
        # Set up the environment
        self.initialize_environment()
    
    def generate_waypoints(self, mode: WaypointGenerationMode = WaypointGenerationMode.OPPOSITE_EDGES) -> List[Waypoint]:
        """
        Generate waypoints according to the specified mode.
        
        Parameters:
        -----------
        mode : WaypointGenerationMode
            Method to use for generating waypoints.
            
        Returns:
        --------
        List[Waypoint]
            Generated waypoints.
        """
        if mode == WaypointGenerationMode.RANDOM:
            return self._generate_random_waypoints()
        elif mode == WaypointGenerationMode.OPPOSITE_EDGES:
            return self._generate_opposite_edges_waypoints()
        else:
            return self._generate_manual_waypoints()
    
    def _generate_random_waypoints(self) -> List[Waypoint]:
        """Generate random waypoints within environment bounds."""
        points = np.random.uniform(
            low=[-self.config.bounds[0], -self.config.bounds[1], 0],
            high=[self.config.bounds[0], self.config.bounds[1], self.config.bounds[2]],
            size=(self.config.num_waypoints - 1, 3)
        )
        return [Waypoint(0, 0, 0)] + [
            Waypoint(x, y, z) for x, y, z in points
        ]
    
    def _generate_opposite_edges_waypoints(self) -> List[Waypoint]:
        """Generate waypoints on opposite edges of the environment."""
        bx, by, bz = self.config.bounds
        
        # Generate first point on edge
        x1 = np.random.choice([-bx, bx])
        y1 = np.random.uniform(-by, by)
        z1 = np.random.uniform(0, bz)
        
        # Generate opposite point
        x2 = -x1
        y2 = np.random.choice([-by, by]) if abs(x1) == bx else np.random.uniform(-by, by)
        z2 = bz - z1
        
        # Generate intermediate points
        middle_points = np.random.uniform(
            low=[-bx/2, -by/2, 0],
            high=[bx/2, by/2, bz/2],
            size=(3, 3)
        )
        
        return [
            Waypoint(x1, y1, z1),
            *[Waypoint(x, y, z) for x, y, z in middle_points],
            Waypoint(x2, y2, z2)
        ]
    
    def _generate_manual_waypoints(self) -> List[Waypoint]:
        """Generate a predefined set of waypoints."""
        return [
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
        ]
    
    def initialize_anchors(self) -> Tuple[List[Anchor], List[Anchor]]:
        """
        Initialize both base (known) and unknown UWB anchors.
        
        Returns:
        --------
        Tuple[List[Anchor], List[Anchor]]
            Base anchors and unknown anchors.
        """
        # Initialize base anchors
        bias_model = BiasModel(0.0951, 1.0049)
        noise_model = NoiseModel(0.2, 0.05, (0.2, 0.3))

        base_anchors = [
            Anchor("0", -1, -1, 0, bias_model, noise_model),
            Anchor("1", 1, 0, 0, bias_model, noise_model),
            Anchor("2", 0, 3, 0, bias_model, noise_model),
            Anchor("3", 0, 6, 0, bias_model, noise_model)
        ]
        
        # Initialize unknown anchors with random placement
        bx, by, bz = self.config.bounds
        unknown_position = np.random.uniform(
            low=[-bx, -by, 0],
            high=[bx, by, 0]
        )
        
        unknown_anchors = [
            Anchor("4", *unknown_position, bias_model, noise_model),
        ]
        
        return base_anchors, unknown_anchors
    
    def initialize_environment(self, waypoint_mode: WaypointGenerationMode = WaypointGenerationMode.OPPOSITE_EDGES):
        """
        Initialize the complete simulation environment.
        
        Parameters:
        -----------
        waypoint_mode : WaypointGenerationMode
            Method to use for generating waypoints.
        """
        # Generate waypoints
        self.waypoints = self.generate_waypoints(waypoint_mode)
        
        # Initialize anchors
        self.base_anchors, self.unknown_anchors = self.initialize_anchors()
        
        # Create trajectory
        self.drone_trajectory = Trajectory(
            speed=self.config.drone_speed,
            dt=self.config.dt
        )
        self.drone_trajectory.construct_trajectory_linear(self.waypoints)
        
        # Set initial drone position
        self.drone_position = np.array(self.waypoints[0].get_coordinates())
    
    def update_drone_position_kinematic(self) -> np.ndarray:
        """
        Update the drone position according to the trajectory.
        
        Returns:
        --------
        np.ndarray
            New drone position [x, y, z].
        """
        if self.drone_progress < len(self.drone_trajectory.spline_x):
            new_position = np.array([
                self.drone_trajectory.spline_x[self.drone_progress],
                self.drone_trajectory.spline_y[self.drone_progress],
                self.drone_trajectory.spline_z[self.drone_progress]
            ])
            self.drone_progress += 1
        else:
            # Reset to start if end is reached
            new_position = np.array([
                self.drone_trajectory.spline_x[0],
                self.drone_trajectory.spline_y[0],
                self.drone_trajectory.spline_z[0]
            ])
            self.drone_progress = 0
        
        self.drone_position = new_position
        return new_position
    
    def get_remaining_waypoints(self) -> List[Waypoint]:
        """
        Get waypoints that haven't been visited yet.
        
        Returns:
        --------
        List[Waypoint]
            Unvisited waypoints.
        """
        current_index = np.argmin([
            np.linalg.norm(self.drone_position - np.array([x, y, z]))
            for x, y, z in zip(
                self.drone_trajectory.spline_x,
                self.drone_trajectory.spline_y,
                self.drone_trajectory.spline_z
            )
        ])
        
        waypoint_indices = [
            np.argmin([
                np.linalg.norm(np.array(wp.get_coordinates()) - np.array([x, y, z]))
                for x, y, z in zip(
                    self.drone_trajectory.spline_x,
                    self.drone_trajectory.spline_y,
                    self.drone_trajectory.spline_z
                )
            ])
            for wp in self.waypoints
        ]
        
        return [
            wp for wp, idx in zip(self.waypoints, waypoint_indices)
            if idx > current_index
        ]