#! /usr/bin/env python3
# encoding: utf-8
import numpy as np
import rospy
from gazebo_swarm_robot_control import SwarmRobot


def main():
    # 初始化节点
    rospy.init_node("swarm_robot_control_angle")
    # 机器人的id
    index = [1, 2, 3, 4, 5]
    # 建立对象
    swarm_robot = SwarmRobot(index)

    conv_th = 0.05  # 角度跟踪的阈值
    MAX_W = 1  # 最大角速度 (rad/s)
    MIN_W = 0.05  # 最小角速度 (rad/s)
    # MAX_V = 0.2  # 最大线速度 (m/s)
    # MIN_V = 0.01  # 最小线速度 (m/s)
    k_w = 0.1  # 期望角速度 = k_w * del_theta, del_theta 为某机器人与其他机器人的角度差
    # k_v = 0.1

    # Laplace matrix
    lap = np.array(
        [
            [4, -1, -1, -1, -1],
            [-1, 4, -1, -1, -1],
            [-1, -1, 4, -1, -1],
            [-1, -1, -1, 4, -1],
            [-1, -1, -1, -1, 4],
        ]
    )

    # 存储机器人当前位姿和与其他机器人位姿差
    cur_x = np.zeros(swarm_robot.robot_num)
    cur_y = np.zeros(swarm_robot.robot_num)
    cur_theta = np.zeros(swarm_robot.robot_num)
    del_x = np.zeros(swarm_robot.robot_num)
    del_y = np.zeros(swarm_robot.robot_num)
    del_theta = np.zeros(swarm_robot.robot_num)

    # 运行直到各个机器人角度相同
    is_conv = False  # 是否到达
    while not is_conv:
        # 获取所有机器人的位姿
        current_robot_pose = swarm_robot.get_robot_poses()
        # 提取角度信息, 赋值给 cur_theta
        cur_theta = np.array(
            [current_robot_pose[i][2] for i in range(swarm_robot.robot_num)]
        )

        # 判断是否到达
        del_theta = -np.dot(lap, cur_theta)
        is_conv = np.all(np.abs(del_theta) <= conv_th)
        if is_conv:
            break

        # 控制机器人运动
        for i in range(swarm_robot.robot_num):
            w = k_w * del_theta[i]
            w = swarm_robot.check_vel(w, MAX_W, MIN_W)
            swarm_robot.move_robot(i, 0, w)
        # 等待一段时间
        rospy.sleep(0.05)

    # 停止所有机器人
    swarm_robot.stop_robots()

    rospy.loginfo("Succeed!")


if __name__ == "__main__":
    main()
