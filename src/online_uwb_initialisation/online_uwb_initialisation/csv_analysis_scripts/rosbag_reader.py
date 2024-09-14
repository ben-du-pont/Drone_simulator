#!/usr/bin/env python


import rosbag
import csv
import os
import re
import rospy
from pathlib import Path
from datetime import datetime
import tf.transformations  # For quaternion rotation
from geometry_msgs.msg import PoseWithCovarianceStamped

package_path = Path(__file__).parent.parent.resolve()
bag_dir = package_path / 'bagfiles'
csv_output_dir = package_path / 'csv_files' / 'drone_halle_bags'

# Tag position on the drone relative to drone cog, in the reference frame x to the right, y to the front, z up
# x = 0.065
# y = -0.010
# z = 0.085



# Define the translation of the UWB tag relative to the drone's CoG
TAG_OFFSET_X = 0.065
TAG_OFFSET_Y = -0.010
TAG_OFFSET_Z = 0.085

# Function to rotate a vector using a quaternion
def rotate_vector_by_quaternion(vector, quaternion):
    # Create a 4D vector from the input vector (x, y, z) with w=0 for quaternion multiplication
    vector_q = [vector[0], vector[1], vector[2], 0]

    # Apply the quaternion rotation: q * v * q^(-1)
    q_inv = tf.transformations.quaternion_inverse(quaternion)
    rotated_vector_q = tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(quaternion, vector_q), q_inv
    )

    # Return only the rotated vector part (x, y, z)
    return rotated_vector_q[:3]

# Function to convert drone position to tag position
def compute_tag_position(drone_position, drone_orientation):
    # Define the translation vector for the tag relative to the drone CoG
    tag_translation = [TAG_OFFSET_X, TAG_OFFSET_Y, TAG_OFFSET_Z]

    # Rotate the translation vector by the drone's orientation
    tag_translation_world = rotate_vector_by_quaternion(tag_translation, drone_orientation)

    # Compute the tag's world position by adding the rotated translation to the drone's position
    tag_position_world = [
        drone_position[0] + tag_translation_world[0],
        drone_position[1] + tag_translation_world[1],
        drone_position[2] + tag_translation_world[2],
    ]

    return tag_position_world

def extract_rosbag_to_csv(bag_file, csv_file):
    # Open the rosbag file
    bag = rosbag.Bag(bag_file)

    # Open the CSV file for writing
    with open(csv_file, mode='w') as file:
        writer = csv.writer(file)
        
        # Write the CSV header
        writer.writerow(['anchor_id', 'range', 'tag_position_x', 'tag_position_y', 'tag_position_z'])

        # Read messages from the topic
        for topic, msg, t in bag.read_messages(topics=['/uwb_message/uwb_range_and_pose']):
            print(type(msg))    
            if True:#isinstance(msg, PoseWithCovarianceStamped):
                print(msg)
                # Extract timestamp
                timestamp = t.to_sec()

                # Extract anchor ID from the frame_id field (assuming format "anchor_<id>")
                anchor_id = msg.header.frame_id.split('_')[-1]  # Extract numeric part of "anchor_<id>"

                # Extract range from the covariance field (assuming it's stored in covariance[0])
                range_measurement = msg.pose.covariance[0]

                # Extract drone position
                position_x = msg.pose.pose.position.x
                position_y = msg.pose.pose.position.y
                position_z = msg.pose.pose.position.z
                drone_position = [position_x, position_y, position_z]

                # Extract drone orientation as a quaternion
                orientation_x = msg.pose.pose.orientation.x
                orientation_y = msg.pose.pose.orientation.y
                orientation_z = msg.pose.pose.orientation.z
                orientation_w = msg.pose.pose.orientation.w
                drone_orientation = [orientation_x, orientation_y, orientation_z, orientation_w]

                # Compute tag position in the world frame
                tag_position_world = compute_tag_position(drone_position, drone_orientation)

                # Write the extracted data as a row in the CSV
                writer.writerow([anchor_id, range_measurement] + tag_position_world)

    # Close the rosbag
    bag.close()
    rospy.loginfo(f"Data extracted to {csv_file} successfully")

def process_bag_files(bag_directory, output_directory):
    # Regular expression to match bag files with trajectory naming convention
    bag_file_pattern = r"trajectory_(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}).bag"
    bag_files = []

    # Iterate through all files in the bag directory and find those matching the pattern
    for filename in os.listdir(bag_directory):
        match = re.match(bag_file_pattern, filename)
        if match:
            trajectory_number = match.group(1)
            datetime_str = match.group(2)
            # Parse the date and time from the filename
            timestamp = datetime.strptime(datetime_str, '%Y-%m-%d-%H-%M-%S')
            bag_file_path = os.path.join(bag_directory, filename)
            bag_files.append((trajectory_number, timestamp, bag_file_path))

    # Sort bag files first by trajectory number, then by timestamp
    bag_files.sort(key=lambda x: (int(x[0]), x[1]))

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Dictionary to track sequence number for each trajectory
    trajectory_counters = {}

    # Process each bag file in order
    for trajectory_number, timestamp, bag_file_path in bag_files:
        # Get the current sequence number for the trajectory
        if trajectory_number not in trajectory_counters:
            trajectory_counters[trajectory_number] = 1
        sequence_number = trajectory_counters[trajectory_number]

        # Create CSV file name based on the trajectory and sequence number
        csv_filename = f"trajectory_{trajectory_number}_{sequence_number}.csv"
        csv_file_path = os.path.join(output_directory, csv_filename)

        # Extract data from the bag file to the CSV
        rospy.loginfo(f"Processing {bag_file_path} -> {csv_filename}")
        extract_rosbag_to_csv(bag_file_path, csv_file_path)

        # Increment the sequence number for the trajectory
        trajectory_counters[trajectory_number] += 1

    print("All bag files processed successfully.")
    rospy.loginfo("All bag files processed successfully.")


if __name__ == '__main__':
    # Set the directory containing your ROS bag files and the output directory for CSVs
    bag_directory = bag_dir
    output_directory = csv_output_dir

    # Call the function to process all bag files
    process_bag_files(bag_directory, output_directory)
