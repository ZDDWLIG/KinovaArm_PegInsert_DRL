import sys
sys.path.append('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it')
import time
import rospy
import moveit_commander
import moveit_msgs.msg
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from math import pi
from control_msgs.msg import *
from trajectory_msgs.msg import *
import actionlib
from std_srvs.srv import Empty
from tf import TransformListener
from robot import Robot
from task import peg_in
from Gen3Env.gen3envc import gen3env
# from gen3env_zuo import gen3env
# from module import TD3,train
import gym
import numpy as np
import csv
import select
import threading
import time

def main(): 
    # Initialize ROS node 

    
    # Create a Robot instance 
    env2 = gym.make(id='peg_in_hole-v0')
    
    # if robot.is_init_success: 
    #     # Continuously retrieve and print Cartesian pose 
    #     rate = rospy.Rate(1) 
    #     # Rate of 1 Hz 
    #     while not rospy.is_shutdown(): 
    #       cartesian_pose = robot.get_cartesian_pose() 
    #       print("Cartesian Pose:") 
    #       print("Position: [x={}, y={}, z={}]".format(cartesian_pose.position.x, cartesian_pose.position.y, cartesian_pose.position.z)) 
    #       print("Orientation: [x={}, y={}, z={}, w={}]".format(cartesian_pose.orientation.x, cartesian_pose.orientation.y, cartesian_pose.orientation.z, cartesian_pose.orientation.w)) 
    #       print("---") 
    #       rate.sleep() 
    #     else: 
    #        print("Robot initialization failed.") 
    
    action = [0,0,0,0]
    state =  [0,0,0,0]
    action_list = []
    state_list = []
    step = 0
    change_peg_state = False
    change_hole_state = False 

    end = 10000

    for step in range(1000):   
        
      # rlist, _, _ = select.select([sys.stdin], [], [], 0)
      # if rlist:
      #   break

      state = action
      action = env2.get_obs()[:4]

      if step != 0:
        print("state:   {}".format(state))
        print("action:  {}".format(action))
        print("--------------------------------")
        
        if (not change_peg_state) and (state[3] > 0.3):
          print("change peg")
          change_peg_state = True

        if (not change_hole_state) and (state[3] < 0.3) and change_peg_state:
          print("change hole")
          change_hole_state = True
          end = step + 100
        
        if change_hole_state and change_peg_state:
          None

        if step == end:
          break

        state_list.append(np.array(state))
        action_list.append(np.array(action))
        # rate.sleep() 

    print("change peg = {}, change hole = {}".format(change_peg_state, change_hole_state))
    np.set_printoptions(linewidth=np.inf)
    state_array = np.array(state_list)
    action_array = np.array(action_list)
    data = [state_array, action_array]

    file_name = "/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/data5.csv"
    with open(file_name, 'w') as file:
      writer = csv.writer(file)
      writer.writerows(data)
      print("save csv")
      print("step={}".format(step))

def task():
  time.sleep(1)
  envt=gym.make(id='peg_in_hole-v1')
  peg_in(robot=env.robot,peg_pose=[0.32,0,0.056],hole_pose=[0.5,-0.1671,0.155])
  # peg_pose=[0,0,0.3] hole_pose=[0.8,-0.5,0.3]


if __name__ == '__main__': 
  env=gym.make(id='peg_in_hole-v1')
  env.reset()
  env.robot.move(pose=[0.3, 0, 0.3], tolerance=0.0001) # 前后 左右 上下
  env.robot.reach_gripper_position(0)

  # thread1 = threading.Thread(target=main)
  thread2 = threading.Thread(target=task)
  # thread1.start()
  thread2.start()
  # thread1.join()
  thread2.join()

# env.robot.move(pose=[0.3, 0, 0.3])
# self.peg_pose=[-0.2,0,0.3]  
# self.hole_pose=[0.7,-0.4,0.3]
# peg_in(robot=env.robot,peg_pose=[0.32,-0.003,0.056],hole_pose=[0.5,-0.1671,0.165])
