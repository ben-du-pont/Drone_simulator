import gymnasium as gym
from gymnasium import spaces, register
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from drone_uwb_simulator.drone_simulator import DroneSimulation

from online_uwb_initialisation.uwb_online_initialisation import UwbOnlineInitialisation




class UWBAnchorEnv(gym.Env):
    """Environment for optimizing drone trajectory to improve anchor position estimation in a single action per episode."""
    def __init__(self):
        super(UWBAnchorEnv, self).__init__()

        self.scaling_max_value = 100.0
        self.max_waypoints = 5  # Number of waypoints to consider for simplicity

        self.max_episode_length = 10000  # Maximum number of steps in an episode

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3 * self.max_waypoints,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3 + 5 + 3 * self.max_waypoints,), dtype=np.float64)

        
        
        # Initialize environment
        self.init_environment()

        # initialise state by computing the rough anchor estimate
        self.initialize_state()

    def init_environment(self):
        self.counter = 0
        self.latest_action = None
        self.fig = None
        self.ax = None

        self.drone_sim = DroneSimulation()
        self.uwb_initialisation = UwbOnlineInitialisation()

        self.drone_position_post_rough_estimate = None

        self.anchor_rough_estimate_linear = [None, None, None, None, None]
        self.anchor_rough_estimate_non_linear = [None, None, None, None, None]

        self.original_anchor_estimator = [None, None, None, None, None]
        self.final_anchor_estimator  = [None, None, None, None, None]

        self.unknown_anchor = self.drone_sim.unknown_anchors[0]
        self.uwb_initialisation.unknown_anchors = self.drone_sim.unknown_anchors


    def get_unkown_anchor_rough_estimate(self):
        self.drone_position_post_rough_estimate = self.uwb_initialisation.compute_anchor_rough_estimate(self.drone_sim.drone_trajectory, self.unknown_anchor)

        if self.drone_position_post_rough_estimate is not None:
            self.anchor_rough_estimate_linear = self.uwb_initialisation.unknown_anchor_measurements[self.unknown_anchor.anchor_ID]["estimator_rough_linear"]
            self.anchor_rough_estimate_non_linear = self.uwb_initialisation.unknown_anchor_measurements[self.unknown_anchor.anchor_ID]["estimator_rough_non_linear"]



    def initialize_state(self):
        
        try:
            self.uwb_initialisation.reset_all_measurements(self.unknown_anchor.anchor_ID)
            self.get_unkown_anchor_rough_estimate()
            
            self.remaining_waypoints = self.drone_sim.get_remaining_waypoints(self.drone_position_post_rough_estimate)
            self.remaining_waypoint_coordinates = np.array([self.remaining_waypoints[i].get_coordinates() for i in range(len(self.remaining_waypoints))])
            self.remaining_waypoint_coordinates = self.remaining_waypoint_coordinates[:self.max_waypoints]
            
            if len(self.remaining_waypoint_coordinates) < self.max_waypoints:
                self.initialize_state()
        except:
            self.initialize_state()

        print("Environment initialised successfully")
        self.state = np.concatenate((np.array(self.drone_position_post_rough_estimate) / self.scaling_max_value, 
                                     np.array(self.anchor_rough_estimate_non_linear) / self.scaling_max_value, 
                                     self.remaining_waypoint_coordinates.flatten() / self.scaling_max_value))



    def step(self, action):
        self.counter = self.counter + 1
        self.latest_action = action 

        reward = self.calculate_reward(action)
        self.uwb_initialisation.reset_measurements_post_rough_initialisation(self.unknown_anchor.anchor_ID)
        done = False
        if self.counter >= self.max_episode_length:
            done = True
            print("Counter reached max episode length")
        return self.state, reward, done, done, {}


    def get_final_anchor_estimate(self, learned_waypoints):
        self.uwb_initialisation.refine_anchor_positions(learned_waypoints, self.unknown_anchor)
        self.final_anchor_estimator = self.uwb_initialisation.unknown_anchor_measurements[self.unknown_anchor.anchor_ID]["estimator"]


    def get_original_anchor_estimate(self):
        self.uwb_initialisation.refine_anchor_positions(self.remaining_waypoint_coordinates, self.unknown_anchor)
        self.original_anchor_estimator = self.uwb_initialisation.unknown_anchor_measurements[self.unknown_anchor.anchor_ID]["estimator"]


    def calculate_reward(self, action):
        # Recalculate trilateration based on modified waypoints

        self.get_original_anchor_estimate()
        self.uwb_initialisation.reset_measurements_post_rough_initialisation(self.unknown_anchor.anchor_ID)

        learned_waypoints = action.reshape(-1, 3) * self.scaling_max_value

        self.get_final_anchor_estimate(learned_waypoints)

        error_original = np.array(self.uwb_initialisation.calculate_estimator_error(0, self.original_anchor_estimator))
        error_final = np.array(self.uwb_initialisation.calculate_estimator_error(0, self.final_anchor_estimator))

        delta_error_normalised = (error_original - error_final)/np.maximum(np.abs(error_original), np.abs(error_final))

        return 10*delta_error_normalised



    def reset(self, seed=None, options=None):
        # Reset to initial state for a new episode
        self.init_environment()
        self.initialize_state()

        return self.state, {}



    def render(self, mode='human'):
        # Visualize the drone's trajectory and actions
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.clear()
        self.ax.set_title('Drone Trajectory and Actions')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Plot the drone's trajectory
        drone_trajectory = np.array([[x,y,z] for (x,y,z) in zip(self.drone_sim.drone_trajectory.spline_x, self.drone_sim.drone_trajectory.spline_y, self.drone_sim.drone_trajectory.spline_z)])
        if len(drone_trajectory) > 0:
            self.ax.plot(drone_trajectory[:, 0], drone_trajectory[:, 1], drone_trajectory[:, 2], label='Drone Trajectory')

        # Plot the unknown anchor position
        unknown_anchor_pos = self.unknown_anchor.get_anchor_coordinates()
        self.ax.scatter(unknown_anchor_pos[0], unknown_anchor_pos[1], unknown_anchor_pos[2], color='r', s=100, label='Unknown Anchor')

        # Plot the waypoints
        waypoint_coords = self.remaining_waypoint_coordinates
        self.ax.scatter(waypoint_coords[:, 0], waypoint_coords[:, 1], waypoint_coords[:, 2], color='g', s=50, label='Waypoints')

        # Plot the latest action (generated waypoints)
        if self.latest_action is not None:
            action_waypoints = self.latest_action.reshape(-1, 3) * self.scaling_max_value
            self.ax.scatter(action_waypoints[:, 0], action_waypoints[:, 1], action_waypoints[:, 2], color='b', s=50, label='Action Waypoints')

        self.ax.legend()
        plt.show()



    def close(self):
        # Perform any cleanup
        pass




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


register(
    id='UWBAnchor-v0',
    entry_point='UWB_sim_RL_environment:UWBAnchorEnv',
    max_episode_steps=1000,
)
