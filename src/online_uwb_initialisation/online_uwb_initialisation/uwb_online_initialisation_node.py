import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from sim_interfaces.msg import StampedFloat, DronePosition
from sim_interfaces.srv import AnchorInfo, TrajectoryInfo
from nav_msgs.msg import Path

from drone_uwb_simulator.UWB_protocol import Anchor
from online_uwb_initialisation.uwb_online_initialisation import UwbOnlineInitialisation
import numpy as np

import tf2_ros


class UwbOnlineInitialisationNode(Node):
    def __init__(self):
        super().__init__('uwb_online_initialisation_node')
        self.uwb_online_initialisation = UwbOnlineInitialisation()

        self.get_anchor_info_client = self.create_client(AnchorInfo, 'get_anchor_info')
        self.get_trajectory_info_client = self.create_client(TrajectoryInfo, 'get_trajectory_info')

        # Wait for the service to become available
        while not self.get_anchor_info_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        # Send a request to get anchor information
        self.anchor_info_request = AnchorInfo.Request()
        self.trajectory_info_request = TrajectoryInfo.Request()

        self.get_anchor_info()
        self.get_trajectory_info()

        self.drone_position_subscription = self.create_subscription(DronePosition, 'drone_position', self.drone_position_callback, 10)

        self.error_publisher = self.create_publisher(StampedFloat, 'error_in_anchor_estimate', 10)
        self.gdop_publisher = self.create_publisher(StampedFloat, 'gdop', 10)
        self.fim_publisher = self.create_publisher(StampedFloat, 'fim', 10)
        self.sum_of_residuals_publisher = self.create_publisher(StampedFloat, 'sum_of_residuals', 10)
        self.condition_number_publisher = self.create_publisher(StampedFloat, 'condition_number', 10)
        self.covariances_publisher = self.create_publisher(PoseStamped, 'covariances', 10)
        self.number_of_measurements_publisher = self.create_publisher(StampedFloat, 'number_of_measurements', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.optimised_waypoints_publisher = self.create_publisher(MarkerArray, 'optimised_waypoints', 10)
        self.optimised_trajectory_publisher = self.create_publisher(Path, 'drone_optimised_trajectory', 10)

    def get_anchor_info(self):
        future = self.get_anchor_info_client.call_async(self.anchor_info_request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()

            self.known_anchor_IDs = response.known_anchor_ids
            self.known_anchor_x_positions = response.known_anchor_x_positions
            self.known_anchor_y_positions = response.known_anchor_y_positions
            self.known_anchor_z_positions = response.known_anchor_z_positions
            self.known_anchor_biases = response.known_anchor_biases
            self.known_anchor_linear_biases = response.known_anchor_linear_biases
            self.known_anchor_noise_variances = response.known_anchor_noise_variances

            self.unknown_anchor_IDs = response.unknown_anchor_ids
            self.unknown_anchor_x_positions = response.unknown_anchor_x_positions
            self.unknown_anchor_y_positions = response.unknown_anchor_y_positions
            self.unknown_anchor_z_positions = response.unknown_anchor_z_positions
            self.unknown_anchor_biases = response.unknown_anchor_biases
            self.unknown_anchor_linear_biases = response.unknown_anchor_linear_biases
            self.unknown_anchor_noise_variances = response.unknown_anchor_noise_variances
            self.get_logger().info('Succesfully got anchor information')
            self.process_anchor_info()
            self.get_logger().info('Succesfully processed anchor information')
        else:
            self.get_logger().error('Failed to get anchor information')
    
    def get_trajectory_info(self):
        future = self.get_trajectory_info_client.call_async(self.trajectory_info_request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()

            self.uwb_online_initialisation.trajectory_waypoints = [[x, y, z] for x, y, z in zip(response.waypoints_x, response.waypoints_y, response.waypoints_z)]
            self.uwb_online_initialisation.optimised_trajectory_waypoints = self.uwb_online_initialisation.trajectory_waypoints

            self.get_logger().info('Succesfully got trajectory information')

        else:
            self.get_logger().error('Failed to get trajectory information')

    def process_anchor_info(self):
        number_of_known_anchors = len(self.known_anchor_x_positions)
        number_of_unknown_anchors = len(self.unknown_anchor_x_positions)

        for i in range(number_of_known_anchors):
            anchor = Anchor(self.known_anchor_IDs[i], self.known_anchor_x_positions[i], self.known_anchor_y_positions[i], self.known_anchor_z_positions[i], self.known_anchor_biases[i], self.known_anchor_linear_biases[i], self.known_anchor_noise_variances[i])
            self.uwb_online_initialisation.base_anchors.append(anchor)
        
        for i in range(number_of_unknown_anchors):
            anchor = Anchor(self.unknown_anchor_IDs[i], self.unknown_anchor_x_positions[i], self.unknown_anchor_y_positions[i], self.unknown_anchor_z_positions[i], self.unknown_anchor_biases[i], self.unknown_anchor_linear_biases[i], self.unknown_anchor_noise_variances[i])
            self.uwb_online_initialisation.unknown_anchors.append(anchor)

    def drone_position_callback(self, msg):
        drone_x = msg.position_x
        drone_y = msg.position_y
        drone_z = msg.position_z

        waypoints_visited = msg.waypoints_achieved

        self.uwb_online_initialisation.drone_postion = [drone_x, drone_y, drone_z]

        self.uwb_online_initialisation.measurement_callback([drone_x, drone_y, drone_z], waypoints_visited)
        
        first_key = next(iter(self.uwb_online_initialisation.unknown_anchor_measurements))

        # error = self.uwb_online_initialisation.calculate_position_error(self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["estimated_position"], self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["estimated_position"])
        error = self.uwb_online_initialisation.error
        
        gdop = self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["GDOP"][-1]
        FIM = self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["FIM"][-1]
        sum_of_residuals = self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["sum_of_residuals"][-1]
        condition_number = self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["condition_number"][-1]
        covariances = self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["covariances"][-1]
        number_of_measurements = len(self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["distances_pre_rough_estimate"]) + len(self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["distances_post_rough_estimate"])
        linear_estimated_position = self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["estimated_position_rough_linear"]
        refined_estimated_position = self.uwb_online_initialisation.unknown_anchor_measurements[first_key]["estimated_position_rough_non_linear"]

        self.publish_unkown_anchor_estimated_pose(linear_estimated_position)
        if len(refined_estimated_position) > 0:
            self.publish_unkown_anchor_estimated_pose_refined(refined_estimated_position)

        self.publish_errors(error)
        self.publish_GDOP(gdop)
        self.publish_FIM(FIM)
        self.publish_sum_of_residuals(sum_of_residuals)
        self.publish_condition_number(condition_number)
        self.publish_covariances(covariances)
        self.publish_number_of_measurements(number_of_measurements)

        self.publish_optimised_trajectory_waypoints()

        if self.uwb_online_initialisation.spline_x is not None:
            self.publish_optimised_spline(self.uwb_online_initialisation.spline_x, self.uwb_online_initialisation.spline_y, self.uwb_online_initialisation.spline_z)
            self.uwb_online_initialisation.spline_x = None

    def publish_errors(self, errors):
        error_msg = StampedFloat()
        error_msg.data = float(errors)
        error_msg.header.stamp = self.get_clock().now().to_msg()
        self.error_publisher.publish(error_msg)

    def publish_optimised_trajectory_waypoints(self):
        marker_array = MarkerArray()

        for i, point in enumerate(self.uwb_online_initialisation.optimised_trajectory_waypoints):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.1  # adjust size as needed
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0  # red color
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = float(point[2])
            marker.id = i
            marker_array.markers.append(marker)

        self.optimised_waypoints_publisher.publish(marker_array)

    def publish_optimised_spline(self, spline_x, spline_y, spline_z):
        self.trajectory_publisher = self.create_publisher(Path, 'drone_optimised_trajectory', 10)
        path_msg = Path()
        path_msg.header.frame_id = 'world'  # Set the frame ID

        # Populate the Path message with PoseStamped messages
        for i in range(len(spline_x)):
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'world'  # Set the frame ID
            pose_msg.header.stamp = self.get_clock().now().to_msg()  # Set the timestamp

            # Set the position from the spline data
            pose_msg.pose.position.x = spline_x[i]
            pose_msg.pose.position.y = spline_y[i]
            pose_msg.pose.position.z = spline_z[i]

            # Append the PoseStamped message to the Path message
            path_msg.poses.append(pose_msg)

        # Publish the Path message
        self.optimised_trajectory_publisher.publish(path_msg)

    def publish_GDOP(self, gdop):
        gdop_msg = StampedFloat()
        gdop_msg.data = float(gdop)
        gdop_msg.header.stamp = self.get_clock().now().to_msg()
        self.gdop_publisher.publish(gdop_msg)

    def publish_FIM(self, fim):
        fim_determinant = np.linalg.det(fim)
        fim_msg = StampedFloat()
        fim_msg.data = float(fim_determinant)
        fim_msg.header.stamp = self.get_clock().now().to_msg()
        self.fim_publisher.publish(fim_msg)

    def publish_sum_of_residuals(self, sum_of_residuals):
        sum_of_residuals_msg = StampedFloat()
        sum_of_residuals_msg.data = float(sum_of_residuals)
        sum_of_residuals_msg.header.stamp = self.get_clock().now().to_msg()
        self.sum_of_residuals_publisher.publish(sum_of_residuals_msg)
    
    def publish_condition_number(self, condition_number):
        condition_number_msg = StampedFloat()
        condition_number_msg.data = float(condition_number)
        condition_number_msg.header.stamp = self.get_clock().now().to_msg()
        self.condition_number_publisher.publish(condition_number_msg)

    def publish_covariances(self, covariances):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'world'  # Set the frame ID
        pose_msg.header.stamp = self.get_clock().now().to_msg()  # Set the timestamp

        # Set the position from the covariances data
        pose_msg.pose.position.x = covariances[0]
        pose_msg.pose.position.y = covariances[1]
        pose_msg.pose.position.z = covariances[2]

        # Publish the PoseStamped message
        self.covariances_publisher.publish(pose_msg)

    def publish_number_of_measurements(self, number_of_measurements):
        number_of_measurements_msg = StampedFloat()
        number_of_measurements_msg.data = float(number_of_measurements)
        number_of_measurements_msg.header.stamp = self.get_clock().now().to_msg()
        self.number_of_measurements_publisher.publish(number_of_measurements_msg)

    def publish_unkown_anchor_estimated_pose(self, estimated_position):

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = f'anchor_estimated_pose'
        tf_msg.transform.translation.x = estimated_position[0]
        tf_msg.transform.translation.y = estimated_position[1]
        tf_msg.transform.translation.z = estimated_position[2]
        tf_msg.transform.rotation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        self.tf_broadcaster.sendTransform(tf_msg)

    def publish_unkown_anchor_estimated_pose_refined(self, estimated_position):

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = f'anchor_estimated_pose_refined'
        tf_msg.transform.translation.x = estimated_position[0]
        tf_msg.transform.translation.y = estimated_position[1]
        tf_msg.transform.translation.z = estimated_position[2]
        tf_msg.transform.rotation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        self.tf_broadcaster.sendTransform(tf_msg)
    




def main(args=None):
    rclpy.init(args=args)
    uwb_online_initialisation_node = UwbOnlineInitialisationNode()
    rclpy.spin(uwb_online_initialisation_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()