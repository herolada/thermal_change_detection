from glob import glob

from setuptools import setup

package_name = 'thermal_change_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, package_name + '.core'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='herolada',
    maintainer_email='herold.adam7@gmail.com',
    description='Revisit-based thermal change detection.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'thermal_change_detection = thermal_change_detection.node:main',
            'offline_eval = thermal_change_detection.offline_eval:main',
            'debug_report = thermal_change_detection.debug_report:main',
        ],
    },
)
