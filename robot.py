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
import numpy as np
import time
import os
import copy
import subprocess


class Robot(object):
  """ExampleMoveItTrajectories"""
  def __init__(self):

    # Initialize the node
    super(Robot, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('example_move_it_trajectories')
    self.reference_frame='base'
    try:
      self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
      if self.is_gripper_present:
        gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
        self.gripper_joint_name = gripper_joint_names[0]
      else:
        gripper_joint_name = ""
      self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

      # Create the MoveItInterface necessary objects
      arm_group_name = "arm"
      self.robot = moveit_commander.RobotCommander("robot_description")
      self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
      self.scene_pub = rospy.Publisher(rospy.get_namespace(), moveit_commander.PlanningScene, queue_size=5)
      self.spawn = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel,persistent=True)
      self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel,persistent=True)
      self.set_oject_position_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState,persistent=True)
      self.get_oject_position_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState,persistent=True)
      self.get_link_position=rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState,persistent=True)
      self.tf = TransformListener()
      self.colors = dict()
      rospy.sleep(1)

      self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
      self.arm_group.set_planner_id('TRRT')
      self.arm_group.set_pose_reference_frame(self.reference_frame)
      self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)
      if self.is_gripper_present:
        gripper_group_name = "gripper"
        self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

      # rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
    except Exception as e:
      print (e)
      self.is_init_success = False
    else:
      self.is_init_success = True
    

  def reach_named_position(self, target):
    arm_group = self.arm_group
    
    # Going to one of those targets
    # rospy.loginfo("Going to named target " + target)
    # Set the target
    arm_group.set_named_target(target)
    # Plan the trajectory
    planned_path1 = arm_group.plan()
    # print("planned_path", planned_path1)
    # Execute the trajectory and block while it's not finished
    return arm_group.execute(planned_path1, wait=True)

  def reach_joint_angles(self, tolerance=0.001,j0=0,j1=0,j2=0,j3=0,j4=0,j5=0,j6=0):
    arm_group = self.arm_group
    success = True

    # Get the current joint positions
    joint_positions = arm_group.get_current_joint_values()
    # rospy.loginfo("Printing current joint positions before movement :")
    # for p in joint_positions: rospy.loginfo(p)

    # Set the goal joint tolerance
    self.arm_group.set_goal_joint_tolerance(tolerance)

    # Set the joint target configuration
    joint_positions[0] += j0
    joint_positions[1] += j1
    joint_positions[2] += j2
    joint_positions[3] += j3
    joint_positions[4] += j4
    joint_positions[5] += j5
    joint_positions[6] += j6

    arm_group.set_joint_value_target(joint_positions)
    
    # Plan and execute in one command
    success &= arm_group.go(wait=True)

    # Show joint positions after movement
    new_joint_positions = arm_group.get_current_joint_values()    
    # rospy.loginfo("Printing current joint positions after movement :")
    # for p in new_joint_positions: rospy.loginfo(p)
    return success

  def get_cartesian_pose(self):
    arm_group = self.arm_group

    # Get the current pose and display it
    pose = arm_group.get_current_pose()
    # rospy.loginfo("Actual cartesian pose is : ")
    # rospy.loginfo(pose.pose)

    return pose.pose


  def reach_gripper_position(self, relative_position):
    gripper_group = self.gripper_group
    
    # We only have to move this joint because all others are mimic!
    gripper_joint = self.robot.get_joint(self.gripper_joint_name)
    gripper_max_absolute_pos = gripper_joint.max_bound()
    gripper_min_absolute_pos = gripper_joint.min_bound()
    try:
      val = gripper_joint.move(relative_position, True)
      return val
    except:
      return False 
    
  def get_gripper_position(self):
    gripper_group = self.gripper_group
    joint_pose=gripper_group.get_current_joint_values()
    return  joint_pose[0]


  #move to pose
  def move(self, pose=[0,0,0], tolerance=0.005):
    arm_group = self.arm_group
    
    arm_group.set_goal_position_tolerance(tolerance)
    current_pose=copy.deepcopy(arm_group.get_current_pose().pose)
    current_pose.position.x=float(pose[0])
    current_pose.position.y=float(pose[1])
    current_pose.position.z=float(pose[2])
    arm_group.set_pose_target(current_pose)
    # rospy.loginfo("Planning and going to the Cartesian Pose")
    return arm_group.go(wait=True)
  
    #move to pose
  def move_add(self, pose=[0,0,0], tolerance=0.005):
    arm_group = self.arm_group
    
    arm_group.set_goal_position_tolerance(tolerance)
    current_pose=arm_group.get_current_pose().pose
    current_pose.position.x+=pose[0]
    current_pose.position.y+=pose[1]
    current_pose.position.z+=pose[2]
    arm_group.set_pose_target(current_pose)
    # rospy.loginfo("Planning and going to the Cartesian Pose")
    return arm_group.go(wait=True)
  

  #initial arm peg and hole
  def init_scene(self,peg_pose=[0.,0.,0.],hole_pose=[0.,0.,0.]):
    self.reset_scene()
    self.reach_named_position('retract')
    # rospy.wait_for_service("gazebo/spawn_sdf_model",timeout=5)
    peg_orientation = Quaternion(1,0,0,0)
    peg_pose=Pose(Point(peg_pose[0],peg_pose[1],peg_pose[2]),peg_orientation)
    peg_sdf_path='/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/new_object/peg/model.sdf'
    # peg_sdf_path='/home/user/model_editor_models/test_box/model.sdf'
    peg_xml= open(peg_sdf_path,'r').read()
    hole_orientation = Quaternion(np.sqrt(2)/2, np.sqrt(2)/2, 0, 0)
    hole_pose=Pose(Point(hole_pose[0],hole_pose[1],hole_pose[2]),hole_orientation)
    hole_sdf_path='/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/new_object/hole_new3/model.sdf'
    hole_xml= open(hole_sdf_path,'r').read()
    
    self.spawn('peg',peg_xml,"",peg_pose,'world')
    self.spawn('hole',hole_xml,"",hole_pose,'world')
    # print('init scene')
    self.have_peg = True
    self.have_hole = True

  def reset_scene(self):
    pass
    # os.system('pkill gzserver')
    # p=subprocess.run('roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  #clean scene
  def remove_scene(self):
    try:
      self.delete_model('peg')
    except Exception as e:
      # print('peg', e)
      print('Fail to delete, start to reset')
      self.reset_scene()
    try:
      self.delete_model('hole')
    except Exception as e:
      print('Fail to delete, start to reset')
      self.reset_scene()
    # print('remove_scene')


  #get arm/peg/hole pose from gazebo
  def get_obj_pose(self,obj_name):
    get_pose=GetModelStateRequest()
    get_pose.model_name=obj_name
    obj_x=self.get_oject_position_service(get_pose).pose.position.x
    obj_y=self.get_oject_position_service(get_pose).pose.position.y
    obj_z=self.get_oject_position_service(get_pose).pose.position.z

    return obj_x,obj_y,obj_z
  

  #get link(finger) pose
  def get_link_pose(self,link_name):
    get_pose=GetLinkStateRequest()
    get_pose.link_name=link_name
    link_x=self.get_link_position(get_pose).link_state.pose.position.x
    link_y=self.get_link_position(get_pose).link_state.pose.position.y
    link_z=self.get_link_position(get_pose).link_state.pose.position.z
    
    return np.array([link_x,link_y,link_z])
  
