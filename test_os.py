import os
import subprocess
# os.system('cd /catkin_workspace && roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85')
p=subprocess.run('roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)