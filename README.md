# KinovaArm PegInsertTask with DRL
Train Kinova Gen3 arm in Gazebo with deep reinforcement learning
# 启动docker
- ### 将tar.gz 解压缩，会生成一个tar包
 ```
tar -zxvf dockerup.tar.gz
 ```
- ### 将tar包生成镜像
 ```
 docker  load  <  dockerup.tar
  ```
- ### docker查看镜像指令
```
docker images
```
- ### 主系统运行(为了容器能显示界面)
```
sudo apt-get install x11-xserver-utils
xhost +
```
- ### 镜像生成容器
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
```
- ### 查看正在运行的容器
```
docker ps
```
- ### 进入容器
```docker exec -ti pytorch /bin/bash```

# 开仿真环境

- ### 运行

 ```
cd catkin_workspace

source devel/setup.bash
roslaunch kortex_driver kortex_driver.launch gripper:=robotiq_2f_85
roslaunch kortex_gazebo spawn_kortex_robot.launch gripper:=robotiq_2f_85

source devel/setup.bash
roslaunch kortex_examples moveit_example.launch
 ```

# 退出容器
- ctrl+p+q 返回主系统

 ```
docker stop pytorch #停止运行容器
docker start pytorch #运行容器
 ```
