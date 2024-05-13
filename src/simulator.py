import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time

from matplotlib.widgets import Button

from UWB_protocol import Anchor
from drone_dynamics import Waypoint, Trajectory

import cProfile
import pstats



# TODO:
# Speed up or implement the animation in a more complex environment/visualisation tool
# Read the papers to check the understanding of the UWB protocol
# Check the initialisation of the unkown anchor's position is done correctly and work on the optimisation


class DroneSimulation:

    def __init__(self):
        self.waypoints = None
        self.base_anchors = None
        self.unknown_anchors = None
        self.drone_trajectory = None
        self.drone_trail_points = []
        self.errors = []
        self.std_devs = []
        self.anchor_lines = []
        self.estimator = [0, 0, 0, 0, 0]
        self.measurements = []
        self.running = True  # State to manage play/pause

    def initialise_environment(self):
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
        self.waypoints = [Waypoint(0,0,0)] + [Waypoint(x, y, z) for (x, y, z) in np.random.random((10,3))*[10, 10, 10] - [5, 5, 0]]

        # Define the base_anchors 
        self.base_anchors = np.array([
            Anchor(-1, -1, 0, bias = 5, linear_bias = 0.1, noise_variance = 0.5),
            Anchor(1, 0, 0, bias = 5, linear_bias = 0.1, noise_variance = 0.5),
            Anchor(0, 3, 0, bias = 5, linear_bias = 0.1, noise_variance = 0.5),
            Anchor(0, 6, 0, bias = 5, linear_bias = 0.1, noise_variance = 0.5)
        ])

        # Define the unknown anchors
        self.unknown_anchors = np.array([
            Anchor(-1, 1, 0, bias = 0.5, linear_bias = 0.8, noise_variance = 0.5)
        ])

        # Create trajectory
        self.drone_trajectory = Trajectory(speed=5, dt=0.01)
        self.drone_trajectory.construct_trajectory(self.waypoints)

    def plot_initial_state(self):

        fig = plt.figure()
        self.fig = fig

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_box_aspect([1,1,1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')


        self.plot_known_anchors(ax1)
        self.plot_unknown_anchors(ax1)
        self.plot_waypoints(ax1)
        

        # Initialize the moving drone point and drone trail
        self.drone_point, = ax1.plot([], [], [], 'ko', label='Drone Position')  # Drone position marker

        self.plot_trajectory(ax1)
        self.drone_trail, = ax1.plot([], [], [], 'b-', label='Drone Trail', alpha = 0.5)  # Drone trail line

        self.estimated_position_marker, = ax1.plot([], [], [], 'rx', markersize=10, label='Estimated Position')

        # Initialize lines for each base and unknown anchor
        for anchor in np.concatenate((self.base_anchors, self.unknown_anchors)):
            line, = ax1.plot([], [], [], 'm--', alpha=0.5)  # Magenta dashed line
            self.anchor_lines.append(line)

        ax1.legend()

        ax2 = fig.add_subplot(222)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Error')
        self.error_line, = ax2.plot([], [], 'r-')  # Initialize error line plot

        # Define a subplot for the standard deviations
        ax3 = fig.add_subplot(224)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Std Dev (X, Y, Z)')
        self.std_dev_lines = {
            'x': ax3.plot([], [], label='Std X')[0],
            'y': ax3.plot([], [], label='Std Y')[0],
            'z': ax3.plot([], [], label='Std Z')[0],
        }
        ax3.legend()

        
        # Add a button for play/pause
        ax_button = plt.axes([0.5 - 0.025, 0.05, 0.05, 0.03])  # Adjust
        self.button = Button(ax_button, 'Pause')
        self.button.on_clicked(self.toggle_animation)

        return ax1, ax2, ax3


    def update_drone_position(self, i):
        # Return the updated drone positions using self.drone_trajectory
        return self.drone_trajectory.spline_x[i], self.drone_trajectory.spline_y[i], self.drone_trajectory.spline_z[i]


    def update_distance_to_anchors(self, x, y, z):
        # Initialise array with the distances to the anchors
        anchors_distances = []
        unkown_anchor_distances = []

        # Fill in the distances array with the distances to the known anchors
        for anchor in self.base_anchors:
            distance = anchor.request_distance(x, y, z)
            anchors_distances.append(distance)

        # Fill in the distances array with the distances to the unknown anchors (for now only one anyways)
        for unknown_anchor in self.unknown_anchors:
            distance = unknown_anchor.request_distance(x, y, z)
            unkown_anchor_distances.append(distance)

        return anchors_distances, unkown_anchor_distances


    def plot_known_anchors(self, ax):
        ax.scatter([a.x for a in self.base_anchors], [a.y for a in self.base_anchors], [a.z for a in self.base_anchors], c='g', marker='o', label='Initialised Anchors')


    def plot_unknown_anchors(self, ax):
        ax.scatter([a.x for a in self.unknown_anchors], [a.y for a in self.unknown_anchors], [a.z for a in self.unknown_anchors], c='b', marker='o', label='Uninitialised Anchors')


    def plot_waypoints(self, ax):
        ax.scatter([a.x for a in self.waypoints], [a.y for a in self.waypoints], [a.z for a in self.waypoints], c='r', marker='o', label='Waypoints')
        

    def plot_trajectory(self, ax):
        # Add trajectory plotting details
        ax.plot(self.drone_trajectory.spline_x, self.drone_trajectory.spline_y, self.drone_trajectory.spline_z, c='orange', label='Drone Path', alpha=0.2)


    def estimate_anchor_position(self, measurements, initial_guess=[0, 0, 0, 0, 0]):
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
        result = least_squares(lambda x0, measurements: loss_function(x0, measurements)[0], initial_guess, args=(measurements,), jac=lambda x0, measurements: loss_function(x0, measurements)[1])

        # Calculate covariance matrix from the Jacobian at the last evaluated point
        J = result.jac
        cov_matrix = np.linalg.pinv(J.T @ J)  # Moore-Penrose pseudo-inverse of J.T @ J

        return result.x, cov_matrix
    
    def update_simulation(self, i, ax1, ax2, ax3):
        new_x, new_y, new_z = self.update_drone_position(i)
        self.drone_trail_points.append((new_x, new_y, new_z))
        anchors_distances, unknown_anchor_distances = self.update_distance_to_anchors(new_x, new_y, new_z)
        self.measurements.append([new_x, new_y, new_z] + unknown_anchor_distances)
        self.estimator, cov_matrix = self.estimate_anchor_position(self.measurements, self.estimator)

        std_devs = np.sqrt(np.diag(cov_matrix))
        self.std_devs.append(std_devs)
        #print("Standard Deviations: X:{}, Y:{}, Z:{}".format(*std_devs))

        # Update the drone position marker
        self.drone_point.set_data_3d([new_x], [new_y], [new_z])

        # Update the drone trail
        trail_x, trail_y, trail_z = zip(*self.drone_trail_points)
        self.drone_trail.set_data_3d(trail_x, trail_y, trail_z)
        for anchor, line in zip(self.base_anchors, self.anchor_lines):
            line.set_data_3d([anchor.x, new_x], [anchor.y, new_y], [anchor.z, new_z])

        # Calculate the error in the estimated position
        actual_pos = np.array([(a.x, a.y, a.z) for a in self.unknown_anchors])
        estimated_pos = np.array(self.estimator[:3])
        error = np.linalg.norm(estimated_pos - actual_pos)
        self.errors.append(error)

        # Update the estimated position marker
        est_x, est_y, est_z = self.estimator[:3]
        if self.estimated_position_marker is None:
            self.estimated_position_marker, = ax1.plot([est_x], [est_y], [est_z], 'gx', markersize=10)
        else:
            self.estimated_position_marker.set_data_3d([est_x], [est_y], [est_z])

        # Update the error plot
        self.error_line.set_data(range(len(self.errors)), self.errors)
        ax2.set_xlim(0, len(self.errors) + 1)
        ax2.set_ylim(0, max(self.errors) * 1.1)
        last_error = self.errors[-1] if self.errors else 0
        ax2.set_ylim(0, last_error * 10)

        frames = range(len(self.errors))
        self.std_dev_lines['x'].set_data(frames, [s[0] for s in self.std_devs])
        self.std_dev_lines['y'].set_data(frames, [s[1] for s in self.std_devs])
        self.std_dev_lines['z'].set_data(frames, [s[2] for s in self.std_devs])
        ax3.set_xlim(0, len(self.errors) + 1)
        ax3.relim()
        ax3.autoscale_view()
        ax3.set_ylim(0, np.median(self.std_devs) * 1.1)

        self.fig.canvas.draw_idle()


    def toggle_animation(self, event):
        self.running = not self.running  # Toggle state between True and False
        if self.running:
            self.button.label.set_text("Pause")
        else:
            self.button.label.set_text("Play")


    def run_simulation(self):
        ax1, ax2, ax3 = self.plot_initial_state()
        i = 0
        while i < len(self.drone_trajectory.spline_x) and plt.fignum_exists(self.fig.number):
            if self.running:
                self.update_simulation(i, ax1, ax2, ax3)
                i += 1
            plt.pause(1e-6)  # Adjust this for the speed of the animation
        plt.show()

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()


    sim = DroneSimulation()
    sim.initialise_environment()
    sim.run_simulation()


    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats("profile.prof")  # Save the stats to a file