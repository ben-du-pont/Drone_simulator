from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    sim_node = Node(
            package='drone_uwb_simulator',
            executable='drone_simulator',
            name='sim',
            output='screen'
        )
    
    error_calculator_node = Node(
            package='online_uwb_initialisation',
            executable='uwb_online_initialisation',
            name='estimator',
            output='screen'
        )
    
    ld.add_action(sim_node)
    ld.add_action(error_calculator_node)

    return ld