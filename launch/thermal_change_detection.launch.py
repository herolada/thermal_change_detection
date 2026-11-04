import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('thermal_change_detection'), 'config', 'params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('image', default_value='/flir_boson/image_raw',
                              description='Thermal image (run the Boson with video_mode:=RAW16)'),
        DeclareLaunchArgument('camera_info', default_value='/flir_boson/camera_info'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        Node(
            package='thermal_change_detection',
            executable='thermal_change_detection',
            name='thermal_change_detection',
            parameters=[LaunchConfiguration('params_file'),
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[('image', LaunchConfiguration('image')),
                        ('camera_info', LaunchConfiguration('camera_info'))],
            output='screen',
        ),
    ])
