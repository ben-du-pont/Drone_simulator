from setuptools import find_packages ,setup
import os
from glob import glob

package_name = 'drone_uwb_simulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        ('share/' + package_name, ['resource/drone_mesh.glb']),
        ('share/' + package_name, ['resource/car_mesh.stl']),
        #(os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='A simple ROS 2 publisher node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_simulator = drone_uwb_simulator.drone_simulator_node:main',
        ],
    },
)
