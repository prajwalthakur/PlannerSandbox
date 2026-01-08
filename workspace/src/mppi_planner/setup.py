from setuptools import find_packages, setup
from  glob import glob
package_name = 'mppi_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),        
        ('share/' + package_name + '/config', glob('config/*.rviz')),        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='prajwalthakur98@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vanilla_mppi_planner_node = mppi_planner.vanilla_mppi.vanilla_mppi_planner_node:main',
            'dyn_risk_mppi_planner_node = mppi_planner.dyn_risk_mppi.dyn_risk_mppi_node:main'
        ],
    },
)
