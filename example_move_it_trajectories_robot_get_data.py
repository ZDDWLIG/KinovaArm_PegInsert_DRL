#!/usr/bin/env python2

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

# Inspired from http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
# Modified by Alexandre Vannobel to test the FollowJointTrajectory Action Server for the Kinova Gen3 robot

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_moveit_trajectories.py __ns:=my_gen3

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
from robot_test import Robot
from task_test import peg_in
from gen3env_test import gen3env
# from module import TD3,train
import gym
import numpy as np
import csv
import select
import threading

def main(): 
    # Initialize ROS node 

    
    # Create a Robot instance 
    env = gen3env()
    # env.reset()

    print('peg in hole')
    
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

    if env.robot.is_init_success: 
      # Continuously retrieve and print Cartesian pose 
      # rate = rospy.Rate(1000) 
      # Rate of 100 Hz 
      start = True
      
      if not start:
          key_input = input("input 1 to start:")
          print(key_input)
          if key_input == "1":
            print("start")
            start == True
      
      while not rospy.is_shutdown():   
          
        # rlist, _, _ = select.select([sys.stdin], [], [], 0)
        # if rlist:
        #   break

        state[:4] = action
        action = env.get_action()

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
          
          if change_hole_state and change_peg_state:
            break

          state_list.append(np.array(state))
          action_list.append(np.array(action))

        step += 1 

        # rate.sleep() 
      
    else: 
      print("Robot initialization failed.") 

    print("change peg = {}, change hole = {}".format(change_peg_state, change_hole_state))
    np.set_printoptions(linewidth=np.inf)
    state_array = np.array(state_list)
    action_array = np.array(action_list)
    data = [state_array, action_array]

    file_name = "/home/prlab/rule_data/data6.csv"
    with open(file_name, 'w') as file:
      writer = csv.writer(file)
      writer.writerows(data)
      print("save csv")
      print("step={}".format(step))

def task():
   arm = Robot()
   peg_in(robot=arm,peg_pose=[0.5,-0.2,0.1],hole_pose=[0.7,0,0.2])


if __name__ == '__main__': 
   env = gen3env()
   env.go_home()
   print("success")
   thread1 = threading.Thread(target=main)
   thread2 = threading.Thread(target=task)
   thread1.start()
   thread2.start()
   thread1.join()
   thread2.join()

   #state:   [0.5761229521610207, 0.002142048010364936, 0.43402919294400644, 0.31494004933302033]
# action:  [0.57612295 0.00214205 0.43402919 0.31494005]