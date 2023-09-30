# KinovaArm PegInsertTask with DRL
Train Kinova Gen3 arm to insert the plug into the board in Gazebo with deep reinforcement learning and imitation learning
# Environmental preparation
We strongly recommend using docker to deploy environments, and we provide our environment zip pack **dockerup.tar.gz**. 
- ### Get the docker environment
 ```
tar -zxvf dockerup.tar.gz
docker  load  <  dockerup.tar
 ```
- ### Get the graphical display package
```
sudo apt-get install x11-xserver-utils
xhost +
```
- ### Activate docker
```
docker run -d \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=unix$DISPLAY \
  -e GDK_SCALE \
  -e GDK_DPI_SCALE \
  --gpus=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --name pytorch \
  dockerup
docker exec -ti pytorch /bin/bash
```
You can also deploy the environment step by step by referring to  [kinova](https://github.com/Kinovarobotics/ros_kortex)

# Train

- ### Go to the workspace directory

 ```
cd /catkin_workspace
 ```
- ### Start gazebo and load simulated arm
Load configuration file and open Gazebo to visualize， if you dont want visualization then add argument **gazebo_gui:=false** （Can't speed up training anyway).
 ```
source devel/setup.bash
roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85 start_rviz:=false
 ```
- ### Start training
Open a new terminal and use the following commands to start training
 ```
source devel/setup.bash
roslaunch kortex_examples moveit_example.launch
 ```


