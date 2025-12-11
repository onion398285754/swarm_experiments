#! /usr/bin/env python3
# encoding: utf-8
import numpy as np
import rospy
from gazebo_swarm_robot_control import SwarmRobot
import sys


def run_consensus(swarm_robot, lap_matrix, topology_name):
    """
    在给定的拓扑结构下运行角度一致性协议。

    Args:
        swarm_robot: SwarmRobot 类的实例
        lap_matrix: 代表图结构的拉普拉斯矩阵
        topology_name: 拓扑结构的名称 (用于日志输出)
    """
    rospy.loginfo(f"--- Running consensus protocol for {topology_name} topology ---")

    conv_th = 0.05  # 角度跟踪的阈值
    MAX_W = 1  # 最大角速度 (rad/s)
    MIN_W = 0.05  # 最小角速度 (rad/s)
    k_w = 0.2  # 期望角速度的增益系数

    # 存储机器人当前角度和角度差
    cur_theta = np.zeros(swarm_robot.robot_num)
    del_theta = np.zeros(swarm_robot.robot_num)

    # 重置机器人姿态以便开始新的测试
    # (注意: 实际中你可能需要一个重置gazebo环境的函数)
    rospy.loginfo("Resetting robot positions for new test...")
    # 这里我们简单地停止机器人，并假设你在两次运行之间手动重置了环境
    swarm_robot.stop_robots()
    rospy.sleep(2) # 等待2秒确保机器人停止

    start_time = rospy.Time.now()

    # 运行直到各个机器人角度相同或超时
    is_conv = False
    timeout = rospy.Duration(60) # 设置60秒超时

    while not rospy.is_shutdown() and (rospy.Time.now() - start_time) < timeout:
        # 获取所有机器人的位姿
        current_robot_pose = swarm_robot.get_robot_poses()
        cur_theta = np.array([pose[2] for pose in current_robot_pose])

        # 计算角度差
        del_theta = -np.dot(lap_matrix, cur_theta)

        # 判断是否收敛
        if np.all(np.abs(del_theta) <= conv_th):
            is_conv = True
            break

        # 控制机器人运动
        for i in range(swarm_robot.robot_num):
            w = k_w * del_theta[i]
            # 检查速度是否在限制范围内
            if abs(w) > MIN_W:
                w = np.clip(w, -MAX_W, MAX_W)
            else:
                w = 0
            swarm_robot.move_robot(i, 0, w)

        rospy.sleep(0.05)

    # 停止所有机器人
    swarm_robot.stop_robots()

    if is_conv:
        rospy.loginfo(f"Convergence achieved for {topology_name} topology!")
        rospy.loginfo(f"Final angles: {np.rad2deg(cur_theta)}")
    else:
        rospy.logwarn(f"Failed to converge for {topology_name} topology within the timeout period.")
    rospy.sleep(2) # 在下一个测试前暂停


def main():
    rospy.init_node("swarm_robot_topology_comparison")
    robot_ids = [1, 2, 3, 4, 5]
    swarm_robot = SwarmRobot(robot_ids)
    num_robots = len(robot_ids)

    # --- 1. 全连接图 (Fully Connected Graph) ---
    # 每个节点都与其他所有节点相连
    lap_fc = (num_robots - 1) * np.eye(num_robots) - (np.ones((num_robots, num_robots)) - np.eye(num_robots))
    run_consensus(swarm_robot, lap_fc, "Fully Connected")

    # --- 2. 环形图 (Ring Graph) ---
    # 节点 i 只与 i-1 和 i+1 相连 (首尾相连)
    lap_ring = 2 * np.eye(num_robots) - np.diag(np.ones(num_robots-1), 1) - np.diag(np.ones(num_robots-1), -1)
    lap_ring[0, -1] = -1
    lap_ring[-1, 0] = -1
    run_consensus(swarm_robot, lap_ring, "Ring")

    # --- 3. 线形图 (Line Graph) ---
    # 节点 i 只与 i-1 和 i+1 相连 (首尾不相连)
    lap_line = 2 * np.eye(num_robots) - np.diag(np.ones(num_robots-1), 1) - np.diag(np.ones(num_robots-1), -1)
    lap_line[0, 0] = 1
    lap_line[-1, -1] = 1
    run_consensus(swarm_robot, lap_line, "Line")

    # --- 4. 非连通图 (Disconnected Graph) ---
    # 例如: {1,2,3} 形成一个子图, {4,5} 形成另一个子图
    lap_disconnected = np.zeros((num_robots, num_robots))
    # 子图1: {1,2,3} 全连接
    lap_disconnected[0:3, 0:3] = 2 * np.eye(3) - (np.ones((3, 3)) - np.eye(3))
    # 子图2: {4,5} 互相连接
    lap_disconnected[3, 3] = 1
    lap_disconnected[4, 4] = 1
    lap_disconnected[3, 4] = -1
    lap_disconnected[4, 3] = -1
    run_consensus(swarm_robot, lap_disconnected, "Disconnected")

    rospy.loginfo("All topology tests finished!")


if __name__ == "__main__":
    main()