实验1
实机运行的时候可能需要更改一下文件名和头文件名
添加的仿真脚本路径：/workspaces/mr_ws/src/gazebo_swarm_robot_pkgs/gazebo_swarm_robot_tb3/scripts/....
task1： 
运行 rosrun gazebo_swarm_robot_tb3 task1_angle_consensus_topologies.py
要运行某个图把其他的图注释掉即可

预计结果：
全连接图: 所有机器人几乎同时、迅速地调整它们的朝向，并很快收敛到同一个角度。
环形图和线形图: 机器人同样会收敛，但速度会明显慢于全连接图。在线形图中，处于中间的机器人会先与邻居达成一致，然后这种一致性像波一样向两端传播。
非连通图: 机器人1、2、3会收敛到一个共同的角度，而机器人4、5会收敛到另一个角度。整个系统无法达成全局一致性，而是形成了两个局部的“一致性簇”。

task2:
运行 /workspaces/mr_ws/src/gazebo_swarm_robot_pkgs/gazebo_swarm_robot_tb3/scripts/task2_link_formulation.py

仿真里现在没碰撞，如果有碰撞调整一下引力斥力安全距离