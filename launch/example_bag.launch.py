from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lane_following_cam',
            executable='lane_detect',
            output='screen',
            parameters=[{
                'raw_image': True, # True for raw image, False for compressed image
                'image_topic': '/camera/color/image_raw'
            }],
        ),
    ])