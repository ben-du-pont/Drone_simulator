import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.animation import FuncAnimation
from scipy.optimize import least_squares

import time

from UWB_protocol import Anchor
from drone_dynamics import Waypoint, Trajectory


import drone_dynamics, drone_controller, visualisation
import signal
import sys
import argparse

import cProfile
import pstats



# Constants
TIME_SCALING = 1.0 # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002 # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005 # seconds
run = True

# TODO:
# Speed up or implement the animation in a more complex environment/visualisation tool
# Read the papers to check the understanding of the UWB protocol
# Check the initialisation of the unkown anchor's position is done correctly and work on the optimisation


def signal_handler(signal, frame):
    global run
    run = False
    quad.stop_thread()
    ctrl.stop_thread()
    print('Stopping')
    sys.exit(0)

# Define the waypoints
waypoints = np.array([
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

waypoints = [Waypoint(0,0,0)] + [Waypoint(x,y,z) for (x,y,z) in (np.random.random((10,3))-0.5)*10]

# Define the base_anchors 
base_anchors = np.array([
    Anchor(-1, -1, 0, bias = 5, linear_bias = 0.1, noise_variance = 0.5),
    Anchor(1, 0, 0, bias = 5, linear_bias = 0.1, noise_variance = 0.5),
    Anchor(0, 3, 0, bias = 5, linear_bias = 0.1, noise_variance = 0.5),
    Anchor(0, 6, 0, bias = 5, linear_bias = 0.1, noise_variance = 0.5)
])

# Define the unknown anchors
unknown_anchors = np.array([
    Anchor(-1, 1, 0, bias = 0.5, linear_bias = 0.8, noise_variance = 0.5)
])

# Create the drone trajectory spline to follow
drone_trajectory = Trajectory(speed=10, dt=0.01)
drone_trajectory.construct_trajectory(waypoints)



# # Define the quadcopter parameters
# QUADCOPTER={'q1':{'position': waypoints[0].get_coordinates(),'orientation':[0,0,0],'L':0.3,'r':0.1,'prop_size':[10,4.5],'weight':1.2}}
# # Construct the drone
# quad = drone_dynamics.Quadcopter(QUADCOPTER)

# # Define the controller parameters
# CONTROLLER_PARAMETERS = {'Motor_limits':[4000,9000],
#                     'Tilt_limits':[-10,10],
#                     'Yaw_Control_Limits':[-900,900],
#                     'Z_XY_offset':500,
#                     'Linear_PID':{'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
#                     'Linear_To_Angular_Scaler':[1,1,0],
#                     'Yaw_Rate_Scaler':0.18,
#                     'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]},
#                     }
# # Construct the controller
# ctrl = drone_controller.Controller_PID_Point2Point(quad.get_state,quad.get_time,quad.set_motor_speeds,params=CONTROLLER_PARAMETERS,quad_identifier='q1')

# # Start the threads
# quad.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
# ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)

# Catch Ctrl+C to stop threads
signal.signal(signal.SIGINT, signal_handler)

# visualisation_object = visualisation.GUI(quads=QUADCOPTER)


###################################################### Plotting ######################################################

# Extract x, y, and z coordinates of the waypoints on the spline
x, y, z = zip(*[(waypoint.x, waypoint.y, waypoint.z) for waypoint in waypoints])

# Extract x, y, and z coordinates of the anchors
x_anchor = [base_anchor.x for base_anchor in base_anchors]
y_anchor = [base_anchor.y for base_anchor in base_anchors]
z_anchor = [base_anchor.z for base_anchor in base_anchors]

x_unknown_anchor = [unknown_anchor.x for unknown_anchor in unknown_anchors]
y_unknown_anchor = [unknown_anchor.y for unknown_anchor in unknown_anchors]
z_unknown_anchor = [unknown_anchor.z for unknown_anchor in unknown_anchors]

# Figure 1 is the simulation
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_box_aspect([1,1,1])

# Add the anchors and waypoints to the plot
ax1.scatter(x, y, z, c='r', marker='o', label='Waypoints')
ax1.scatter(x_anchor, y_anchor, z_anchor, c='g', marker='o', label='Intialised Anchors')
ax1.scatter(x_unknown_anchor, y_unknown_anchor, z_unknown_anchor, c='b', marker='o', label='Uninistialised Anchors')

# Add the trajectory spline to the plot
ax1.plot(drone_trajectory.spline_x, drone_trajectory.spline_y, drone_trajectory.spline_z, label='Cubic Spline', alpha=0.3)

# Animations
point, = ax1.plot([], [], [], 'ko') # Drone position
point2, = ax1.plot([], [], [], 'ro') # lookahead point
lines = [ax1.plot([], [], [], 'r-', linewidth=0.5, alpha=0.5)[0] for _ in range(len(base_anchors))] # lines as distance to the anchors
drone_trail, = ax1.plot([], [], [], label='Drone trail')
drone_trail_points = []

# Ax1 settings
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# Figure 2 is the error plot
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Frame')
ax2.set_ylabel('Error')

# Animations
error_line, = ax2.plot([], [], 'r-')
errors = [] # Array keeping track of the errors


def estimate_anchor_position(measurements, initial_guess = [0, 0, 0, 0, 0]):
    def loss_function(x0, measurements):
        errors = []
        guess = x0[0:3]
        bias = x0[3]
        linear_bias = x0[4]
        for measurement in measurements:
            x, y, z, measured_dist = measurement[0:4]
            estimated_dist = np.linalg.norm([guess[0] - x, guess[1] - y, guess[2] - z])*(1+linear_bias) + bias #+ np.random.normal(0, variance)
            errors.append(estimated_dist - measured_dist)
        return errors

    result = least_squares(loss_function, initial_guess, args=(measurements,), bounds=([-np.inf, -np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]))
    
    return result.x


# Define the initialization function for the animation
def init():
    global estimator, measurements

    # Initialize list to store previous estimates
    estimator = [0,0,0,0,0]
    measurements = []

    # quad.set_position('q1', waypoints[0].get_coordinates())
    # ctrl.update_target((waypoints[-1].x, waypoints[-1].y, waypoints[-1].z))
    # ctrl.update_yaw_target(0)
    drone_trail_points.clear()
    errors.clear()

    point.set_data([], [])
    point.set_3d_properties([])

    point2.set_data([], [])
    point2.set_3d_properties([])

    drone_trail.set_data([],[])
    drone_trail.set_3d_properties([])

    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return [point, point2] + lines + [drone_trail],

def update_drone_position(i):
    # Get the position of the drone and extract the point to track from the lookahead distance
    x,y,z = quad.get_position('q1')
    new_tracking_point_x, new_tracking_point_y, new_tracking_point_z = drone_trajectory.get_lookahead_distance_waypoint([x,y,z], 0.9).get_coordinates()
    
    # Update the controller inputs to track the lookahead point
    ctrl.update_target((new_tracking_point_x,new_tracking_point_y,new_tracking_point_z))

    # Update the plot with the new position of the drone
    new_x, new_y, new_z = quad.get_position('q1')

    return new_tracking_point_x, new_tracking_point_y, new_tracking_point_z, new_x, new_y, new_z

def update_drone_position_kinematic(i):
    return drone_trajectory.spline_x[i], drone_trajectory.spline_y[i], drone_trajectory.spline_z[i], drone_trajectory.spline_x[i], drone_trajectory.spline_y[i], drone_trajectory.spline_z[i]

def update_distance_to_anchors(x,y,z):
    # Initialise array with the distances to the anchors
    anchors_distances = []
    unkown_anchor_distances = []

    # Fill in the distances array with the distances to the known anchors
    for anchor in base_anchors:
        distance = anchor.request_distance(x, y, z)
        anchors_distances.append(distance)

    # Fill in the distances array with the distances to the unknown anchors (for now only one anyways)
    for unknown_anchor in unknown_anchors:
        distance = unknown_anchor.request_distance(x, y, z)
        unkown_anchor_distances.append(distance)

    return anchors_distances, unkown_anchor_distances


def update_plot(i):
    global estimator, measurements

    new_tracking_point_x, new_tracking_point_y, new_tracking_point_z, new_x, new_y, new_z = update_drone_position_kinematic(i) #update_drone_position_kinematic(i)
    drone_trail_points.append([new_x, new_y, new_z])

    x_trail, y_trail, z_trail = (zip(*drone_trail_points))
    drone_trail.set_data(x_trail, y_trail)
    drone_trail.set_3d_properties(z_trail)

    # update the new point to track in the visualisation
    point2.set_data(new_tracking_point_x,new_tracking_point_y)
    point2.set_3d_properties(new_tracking_point_z)
    
    point.set_data(new_x,new_y)
    point.set_3d_properties(new_z)

    # Update the visual of the lines to the anchors
    for j, line in enumerate(lines):
        if j < len(base_anchors):
            line.set_data([base_anchors[j].x, new_x], [base_anchors[j].y, new_y])
            line.set_3d_properties([base_anchors[j].z, new_z])

    
    anchors_distances, unkown_anchor_distances = update_distance_to_anchors(new_x, new_y, new_z)
    measurements.append([new_x, new_y, new_z]+ unkown_anchor_distances)
    estimator = estimate_anchor_position(measurements, estimator)
    # print(estimator)

    # Calculate error in estimated position of the unkown anchor and keep it in history
    error = np.linalg.norm(estimator[0:3] - np.array([unknown_anchors[0].x, unknown_anchors[0].y, unknown_anchors[0].z]))
    errors.append(error)

    # Update the subplot with the new error value
    error_line.set_data(np.arange(len(errors)), errors)
    ax2.set_xlim(0, len(errors))
    ax2.set_ylim(0, max(errors)* 1.1)


def main():
    init()
    i = 0
    try:
        while True:  # Or some other condition for your simulation
            # Your logic to update x, y, z coordinates
            # Example with placeholders:
            
            # Update the plot with the new data
            update_plot(i)
            i += 1

            plt.pause(0.0001)  # Short pause to allow for GUI events and screen updates
            time.sleep(0.0005)  # Adjust timing based on your simulation needs

    except KeyboardInterrupt:
        print("Stopped by User")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats("profile.prof")  # Save the stats to a file




