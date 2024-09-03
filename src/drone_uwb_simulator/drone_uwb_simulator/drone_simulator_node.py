import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Quaternion, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

from sim_interfaces.srv import AnchorInfo, TrajectoryInfo
from sim_interfaces.msg import DronePosition

import tf2_ros

import threading

from drone_uwb_simulator.drone_simulator import DroneSimulation

class DroneSimulationNode(Node):
    def __init__(self):
        super().__init__('drone_simulation')

        # Sevices to give the anchor and trajectory information to the estimator
        self.get_anchor_info_service = self.create_service(AnchorInfo, 'get_anchor_info', self.get_anchor_info_callback)
        self.get_trajectory_info_service = self.create_service(TrajectoryInfo, 'get_trajectory_info', self.get_trajectory_info_callback)

        # Publishers
        self.drone_position_publisher = self.create_publisher(DronePosition, 'drone_position', 10)
        self.trajectory_publisher = self.create_publisher(Path, 'drone_trajectory', 10)
        self.trail_publisher = self.create_publisher(Path, 'drone_trail', 10)
        self.drone_mesh_publisher = self.create_publisher(Marker, 'drone_marker', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.waypoints_publisher = self.create_publisher(MarkerArray, 'waypoints', 10)

        # Subscribers
        self.optimised_trajectory_subscription = self.create_subscription(Path,'drone_optimised_trajectory', self.optimised_trajectory_callback, 10)

        # Create the drone simulation object
        self.drone_simulation = DroneSimulation(drone_speed=3)

        # Fill the anchor information to be sent to the estimator in the service
        self.fill_anchor_info()


    def fill_anchor_info(self):

        # Initialize anchor positions and parameters
        self.known_anchor_IDs = []
        self.known_anchor_x_positions = []  # List of x positions
        self.known_anchor_y_positions = []  # List of y positions
        self.known_anchor_z_positions = []  # List of z positions
        self.known_anchor_biases = []  # List of biases
        self.known_anchor_linear_biases = []  # List of linear biases
        self.known_anchor_noise_variances = []  # List of noise variances

        # Initialize anchor positions and parameters
        self.unknown_anchor_IDs = []
        self.unknown_anchor_x_positions = []  # List of x positions
        self.unknown_anchor_y_positions = []  # List of y positions
        self.unknown_anchor_z_positions = []  # List of z positions
        self.unknown_anchor_biases = []  # List of biases
        self.unknown_anchor_linear_biases = []  # List of linear biases
        self.unknown_anchor_noise_variances = []  # List of noise variances

        for anchor in self.drone_simulation.base_anchors:
            self.known_anchor_IDs.append(anchor.anchor_ID)
            self.known_anchor_x_positions.append(anchor.x)
            self.known_anchor_y_positions.append(anchor.y)
            self.known_anchor_z_positions.append(anchor.z)
            self.known_anchor_biases.append(anchor.bias)
            self.known_anchor_linear_biases.append(anchor.linear_bias)
            self.known_anchor_noise_variances.append(anchor.noise_variance)

        for anchor in self.drone_simulation.unknown_anchors:
            self.unknown_anchor_IDs.append(anchor.anchor_ID)
            self.unknown_anchor_x_positions.append(anchor.x)
            self.unknown_anchor_y_positions.append(anchor.y)
            self.unknown_anchor_z_positions.append(anchor.z)
            self.unknown_anchor_biases.append(anchor.bias)
            self.unknown_anchor_linear_biases.append(anchor.linear_bias)
            self.unknown_anchor_noise_variances.append(anchor.noise_variance)

    def get_anchor_info_callback(self, request, response):
        response.known_anchor_ids = self.known_anchor_IDs
        response.known_anchor_x_positions = self.known_anchor_x_positions
        response.known_anchor_y_positions = self.known_anchor_y_positions
        response.known_anchor_z_positions = self.known_anchor_z_positions
        response.known_anchor_biases = self.known_anchor_biases
        response.known_anchor_linear_biases = self.known_anchor_linear_biases
        response.known_anchor_noise_variances = self.known_anchor_noise_variances

        response.unknown_anchor_ids = self.unknown_anchor_IDs
        response.unknown_anchor_x_positions = self.unknown_anchor_x_positions
        response.unknown_anchor_y_positions = self.unknown_anchor_y_positions
        response.unknown_anchor_z_positions = self.unknown_anchor_z_positions
        response.unknown_anchor_biases = self.unknown_anchor_biases
        response.unknown_anchor_linear_biases = self.unknown_anchor_linear_biases
        response.unknown_anchor_noise_variances = self.unknown_anchor_noise_variances

        self.get_logger().info('Sending anchor information')

        return response
    
    def get_trajectory_info_callback(self, request, response):
        
        response.waypoints_x = [float(point.x) for point in self.drone_simulation.waypoints]
        response.waypoints_y = [float(point.y) for point in self.drone_simulation.waypoints]
        response.waypoints_z = [float(point.z) for point in self.drone_simulation.waypoints]

        self.get_logger().info('Sending trajectory information')

        return response
    
    


    def run_simulation(self):
        """ Main loop for the simulation."""

        rate = self.create_rate(1/self.drone_simulation.dt)  # Publish rate 

        self.drone_trajectory_initial = self.drone_simulation.drone_trajectory.spline_x, self.drone_simulation.drone_trajectory.spline_y, self.drone_simulation.drone_trajectory.spline_z
        
        while rclpy.ok():
            i = 0
            self.publish_trajectory(*self.drone_trajectory_initial)
            self.publish_anchors_tf()
            self.publish_trajectory_waypoints()

            
            # Update drone position
            new_x, new_y, new_z = self.drone_simulation.update_drone_position_kinematic()
            waypoints_achieved = len(self.drone_simulation.waypoints) - len(self.drone_simulation.get_remaining_waypoints([new_x, new_y, new_z]))


            # Publish drone position
            msg = DronePosition()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            msg.position_x = new_x
            msg.position_y = new_y
            msg.position_z = new_z
            msg.waypoints_achieved = waypoints_achieved
            self.drone_position_publisher.publish(msg)

            # Publish TF data
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = 'world'
            tf_msg.child_frame_id = 'drone'
            tf_msg.transform.translation.x = new_x
            tf_msg.transform.translation.y = new_y
            tf_msg.transform.translation.z = new_z
            tf_msg.transform.rotation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
            self.tf_broadcaster.sendTransform(tf_msg)

            self.publish_drone_marker()

            self.publish_drone_trail(self.drone_simulation.drone_trajectory.spline_x[:i], self.drone_simulation.drone_trajectory.spline_y[:i+1], self.drone_simulation.drone_trajectory.spline_z[:i])

            i += 1
            rate.sleep()

            

            


    def publish_anchors_tf(self):
        # Publish TF frames for each UWB anchor
        for idx, anchor in enumerate(self.drone_simulation.base_anchors):
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = 'world'
            tf_msg.child_frame_id = f'anchor_{anchor.anchor_ID}'
            tf_msg.transform.translation.x = float(anchor.x)
            tf_msg.transform.translation.y = float(anchor.y)
            tf_msg.transform.translation.z = float(anchor.z)
            tf_msg.transform.rotation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
            self.tf_broadcaster.sendTransform(tf_msg)

        for idx, anchor in enumerate(self.drone_simulation.unknown_anchors):
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = 'world'
            tf_msg.child_frame_id = f'unknown_anchor_{anchor.anchor_ID}'
            tf_msg.transform.translation.x = float(anchor.x)
            tf_msg.transform.translation.y = float(anchor.y)
            tf_msg.transform.translation.z = float(anchor.z)
            tf_msg.transform.rotation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
            self.tf_broadcaster.sendTransform(tf_msg)

    def publish_trajectory(self, spline_x, spline_y, spline_z):
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
        self.trajectory_publisher.publish(path_msg)

    def publish_drone_trail(self, spline_x, spline_y, spline_z):
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
        self.trail_publisher.publish(path_msg)

    def publish_drone_marker(self):
        # Create a new marker message
        marker_msg = Marker()
        marker_msg.header.frame_id = 'drone'
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.ns = 'drone_marker'
        marker_msg.id = 0
        marker_msg.type = Marker.MESH_RESOURCE
        marker_msg.action = Marker.ADD
        marker_msg.pose.position.x = 0.0  # Set the position as needed
        marker_msg.pose.position.y = 0.25
        marker_msg.pose.position.z = 0.0
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = 1.0  # Set the scale as needed
        marker_msg.scale.y = 1.0
        marker_msg.scale.z = 1.0
        marker_msg.color.r = 1.0
        marker_msg.color.g = 1.0
        marker_msg.color.b = 1.0
        marker_msg.color.a = 1.0
        marker_msg.lifetime.sec = 0
        marker_msg.lifetime.nanosec = 0
        marker_msg.frame_locked = False
        marker_msg.points = []
        marker_msg.colors = []
        # marker_msg.mesh_file.filename = 'package://resource/drone_mesh.glb'
        # marker_msg.mesh_resource = 'package://drone_uwb_simulator/car_mesh.stl'
        marker_msg.mesh_resource = 'package://drone_uwb_simulator/drone_mesh.glb'
        marker_msg.mesh_use_embedded_materials = True

        # Publish the marker
        self.drone_mesh_publisher.publish(marker_msg)

    def publish_trajectory_waypoints(self):
        marker_array = MarkerArray()

        for i, point in enumerate(self.drone_simulation.waypoints):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.1  # adjust size as needed
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0  # red color
            marker.pose.position.x = float(point.x)
            marker.pose.position.y = float(point.y)
            marker.pose.position.z = float(point.z)
            marker.id = i
            marker_array.markers.append(marker)

        self.waypoints_publisher.publish(marker_array)


    def optimised_trajectory_callback(self, msg):
        self.optimised_spline_x = []
        self.optimised_spline_y = []
        self.optimised_spline_z = []

        for pose_msg in msg.poses:
            self.optimised_spline_x.append(pose_msg.pose.position.x)
            self.optimised_spline_y.append(pose_msg.pose.position.y)
            self.optimised_spline_z.append(pose_msg.pose.position.z)

        self.drone_simulation.drone_trajectory.spline_x = self.optimised_spline_x
        self.drone_simulation.drone_trajectory.spline_y = self.optimised_spline_y
        self.drone_simulation.drone_trajectory.spline_z = self.optimised_spline_z

        self.drone_simulation.drone_progress = 0




def main(args=None):
    rclpy.init(args=args)
    sim = DroneSimulationNode()
    thread = threading.Thread(target=rclpy.spin, args=(sim, ), daemon=True)
    thread.start()

    sim.run_simulation()
    rclpy.shutdown()

if __name__ == '__main__':
    main()