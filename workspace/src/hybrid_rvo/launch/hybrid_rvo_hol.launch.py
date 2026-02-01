import os

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():


    pkg_name = 'hybrid_rvo'
    pkg_share = get_package_share_directory(pkg_name)
    sim_config = LaunchConfiguration('sim_config')
    hol_robot_sdf = LaunchConfiguration('hol_robot_sdf')
    
    # --- RViz config ---
    rviz_config = os.path.join(pkg_share,'config','mppi_rviz.rviz')
    # --- RViz ---
    rviz = Node(package='rviz2',executable='rviz2',output='screen',arguments=['-d', rviz_config],)
    gazebo_ros_pkg = get_package_share_directory('gazebo_ros')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'verbose': 'true'
        }.items()
    )


    # --- holonomic robots ---
    declare_sim_config = DeclareLaunchArgument(
        'sim_config',
        default_value=os.path.join(
            pkg_share, 'config', 'sim_config.yaml'),
        description='path to sim config'
    )
    declare_hol_robot_sdf = DeclareLaunchArgument(
        'hol_robot_sdf',
        default_value=os.path.join(
            pkg_share, 'robot_model', 'hol_robot.sdf'),
        description='Path to robot SDF'
    )
    spawn_hol_robots = Node(
        package= pkg_name,
        executable='spawn_hol_robot',
        name='spawn_hol_robot',
        output='screen',
        arguments=[
                sim_config,
                hol_robot_sdf
            ]
    )
    return LaunchDescription([
        declare_sim_config,
        declare_hol_robot_sdf,
        gazebo,
        rviz,
        spawn_hol_robots,
    ])
