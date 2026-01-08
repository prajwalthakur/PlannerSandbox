import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # --- RViz config ---
    rviz_config = os.path.join(
        get_package_share_directory('mppi_planner'),
        'config',
        'mppi_rviz.rviz'
    )

    # --- TurtleBot3 empty world (spawns robot automatically) ---
    turtlebot3_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'launch',
                'empty_world.launch.py'
            )
        ),
        # launch_arguments={
        #     'params_file': os.path.expanduser(
        #         '/root/workspace/src/obstacle_sim/config/diff_drive_limits.yaml'
        #     )
        # }.items()
    )
    # --- RViz ---
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    # --- Cylindrical obstacles ---
    spawn_cylinders = Node(
        package='obstacle_sim',
        executable='spawn_cylinder.py',
        output='screen',
    )

    return LaunchDescription([
        turtlebot3_world,
        rviz,
        spawn_cylinders,
    ])
