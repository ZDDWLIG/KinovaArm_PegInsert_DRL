import sys
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
import time

def pick_place(robot,pick_pose=[0,0,0],place_pose=[0,0,0],tolerance=0.005,joint_rota=0,success=True):
    rospy.loginfo('Execute pick and place task...')
    rospy.loginfo('Go back to initial pose')
    success &= robot.reach_named_position("retract")
    success&=robot.reach_joint_angles(j5=-pi/36)
    print(success)

    success &= robot.move(pose=[pick_pose[0],pick_pose[1],0])
    success &= robot.move(pose=[0,0,pick_pose[2]])
    success&=robot.reach_joint_angles(j6=joint_rota)
    success &= robot.reach_gripper_position(0)
    rospy.loginfo('Arrived object pose, prepare for grabing...')
    print(success)
    success&=robot.reach_gripper_position(0.92)
    rospy.loginfo('Go to target pose')
    success &= robot.move(pose=[0,0,0.2])
    success &= robot.move(pose=[place_pose[0],place_pose[1],0])
    success &= robot.move(pose=[0,0,place_pose[2]])
    success &= robot.move(pose=[0,0,-0.2])
    success &= robot.reach_gripper_position(0)
    print(success)
    rospy.loginfo('Task finished, back to home')
    success &= robot.reach_named_position("home")
    print(success)
    return success
def screw(robot,nut_pose,target_pose,tolerance=0.001,success=True):
   rospy.loginfo('Execute screw task...')
   rospy.loginfo('Go back to initial pose')
   success &= robot.reach_named_position("retract")
   success&=robot.reach_joint_angles(j5=-pi/36)
   print(success)
   success &= robot.move(pose=[nut_pose[0],nut_pose[1],nut_pose[2]],tolerance=tolerance)
   success &= robot.reach_gripper_position(0)
   rospy.loginfo('Arrive object pose, prepare for grabing nut...')
   print(success)
   success&=robot.reach_gripper_position(0.9)
   rospy.loginfo('Go to target pose')
   success &= robot.move(pose=[target_pose[0],target_pose[1],target_pose[2]],tolerance=tolerance)
   success&=robot.move(pose=[0,0,-0.045],tolerance=tolerance)
   rospy.loginfo('Screwing')
   success&=robot.reach_joint_angles(j6=pi)
   print(success)
   rospy.loginfo('Task finished, back to home')
   success &= robot.reach_gripper_position(0) 
   success &= robot.reach_named_position("home")
   return success

 
def peg_in(robot,peg_pose,hole_pose,tolerance=0.001,success=True):
   rospy.loginfo('Execute peg in hole tast...')
   rospy.loginfo('Go back to initial pose')
  #  success &= robot.reach_named_position("retract")
  #  print(joint_positions)geometry_msgs.msg.
   # success&=robot.reach_named_position('retract')
   # success&=robot.reach_joint_angles(j5=-pi/36)
   # print(success)
   success&=robot.reach_gripper_position(0)
   success&=robot.move(pose=[peg_pose[0],peg_pose[1],peg_pose[2]], tolerance=0.0001)
   
   rospy.loginfo('Arrive peg pose, perpare for grabing peg...')
   success&=robot.reach_gripper_position(0.465)#465
   # time.sleep(1)
   print('next')
   success&=robot.move(pose=[peg_pose[0],peg_pose[1],peg_pose[2]+0.1], tolerance=0.001)
  #  success &= robot.reach_named_position("retract")
  #  success &= robot.reach_joint_angles(j5=-pi/36)
   rospy.loginfo('Start to peg in...')
  #  success&=robot.reach_joint_angles(j5=pi/2)
   success&=robot.move(pose=[hole_pose[0],hole_pose[1],hole_pose[2]], tolerance=0.001)
   x = hole_pose[0]+0.085
   time.sleep(1)
   success&=robot.move(pose=[x,hole_pose[1],hole_pose[2]], tolerance=0.001)
   success&=robot.reach_gripper_position(0)
   time.sleep(0.5)
   # success&=robot.move(pose=[0.06,0,0],tolerance=tolerance)
   print(success)
   return success
