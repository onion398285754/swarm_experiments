#! /usr/bin/env python3
# encoding: utf-8
import numpy as np
import rospy
from gazebo_swarm_robot_control import SwarmRobot


def main():
    # 初始化节点
    rospy.init_node("swarm_robot_control_line_collision_avoidance")
    
    # 机器人的id列表
    index = [1, 2, 3, 4, 5]
    robot_num = len(index)
    
    # 建立对象
    swarm_robot = SwarmRobot(index)

    # === 参数设置 ===
    # 阈值
    conv_pos = 0.05
    conv_ang = 0.05
    
    # 速度限制
    MAX_W = 1.0
    MIN_W = 0.05
    MAX_V = 0.2
    MIN_V = 0.01

    # 控制增益
    k_v = 1.0
    k_w_ang = 0.5
    k_w_pos = 1.5

    # === 避障参数 (新增核心部分) ===
    # 检测范围：如果距离小于该值，开始产生斥力
    DETECT_DIST = 0.8  
    # 斥力增益：数值越大，推开的力度越大
    K_REP = 0.8        

    # 拉普拉斯矩阵
    lap = np.array(
        [
            [4, -1, -1, -1, -1],
            [-1, 4, -1, -1, -1],
            [-1, -1, 4, -1, -1],
            [-1, -1, -1, 4, -1],
            [-1, -1, -1, -1, 4],
        ]
    )

    # === 编队形状定义 ===
    dist_interval = 0.8
    offset_x = np.array([0, 1, 2, 3, 4]) * dist_interval
    offset_y = np.zeros(robot_num)

    # 状态变量
    cur_x = np.zeros(robot_num)
    cur_y = np.zeros(robot_num)
    cur_theta = np.zeros(robot_num)
    
    rospy.loginfo("Start Linear Formation Control with Collision Avoidance...")

    is_conv = False
    while not is_conv and not rospy.is_shutdown():
        # 1. 获取位姿
        current_robot_pose = swarm_robot.get_robot_poses()
        for i in range(robot_num):
            cur_x[i] = current_robot_pose[i][0]
            cur_y[i] = current_robot_pose[i][1]
            cur_theta[i] = current_robot_pose[i][2]

        # 2. 计算一致性引力 (Formation Force)
        u_x_formation = -np.dot(lap, cur_x - offset_x)
        u_y_formation = -np.dot(lap, cur_y - offset_y)
        u_theta_global = -np.dot(lap, cur_theta)

        # === 3. 计算避障斥力 (Repulsive Force) ===
        u_x_rep = np.zeros(robot_num)
        u_y_rep = np.zeros(robot_num)

        for i in range(robot_num):
            for j in range(robot_num):
                if i == j:
                    continue
                
                # 计算机器人 i 和 j 的距离
                dx = cur_x[i] - cur_x[j]
                dy = cur_y[i] - cur_y[j]
                dist = np.sqrt(dx**2 + dy**2)

                # 如果距离太近，产生斥力
                if dist < DETECT_DIST:
                    # 斥力公式：力的大小与距离的平方成反比 (或者 1/dist - 1/detect_dist)
                    # 这是一个简单的反比例斥力场
                    rep_force = K_REP * (1.0 / dist - 1.0 / DETECT_DIST) / (dist**2)
                    
                    # 限制最大斥力，防止速度由于距离极近而爆炸
                    if rep_force > 2.0:
                        rep_force = 2.0

                    # 向量分解：斥力方向沿着连线向外 (dx, dy 已经是 i - j，即指向外的向量)
                    u_x_rep[i] += rep_force * dx
                    u_y_rep[i] += rep_force * dy

        # 4. 合成速度向量 (Total = Formation + Repulsion)
        u_x_total = u_x_formation + u_x_rep
        u_y_total = u_y_formation + u_y_rep

        # 5. 判断收敛条件 (只看编队误差，忽略斥力引起的临时误差)
        pos_err_norm = np.sqrt(u_x_formation**2 + u_y_formation**2)
        if np.all(pos_err_norm < conv_pos) and np.all(np.abs(u_theta_global) < conv_ang):
            # 还需要判断是否已经没有危险的斥力作用了（即大家都分开了）
            if np.all(np.abs(u_x_rep) < 0.01) and np.all(np.abs(u_y_rep) < 0.01):
                is_conv = True
                rospy.loginfo("Converged!")
                break

        # 6. 运动学解算与控制
        for i in range(robot_num):
            theta = cur_theta[i]
            
            # 使用合成后的 u_total 进行转换
            v_cmd = k_v * (u_x_total[i] * np.cos(theta) + u_y_total[i] * np.sin(theta))
            y_err_body = -u_x_total[i] * np.sin(theta) + u_y_total[i] * np.cos(theta)

            w_cmd = k_w_ang * u_theta_global[i] + k_w_pos * y_err_body

            # 速度限幅
            v_cmd = swarm_robot.check_vel(v_cmd, MAX_V, MIN_V)
            w_cmd = swarm_robot.check_vel(w_cmd, MAX_W, MIN_W)

            swarm_robot.move_robot(i, v_cmd, w_cmd)

        rospy.sleep(0.05)

    swarm_robot.stop_robots()
    rospy.loginfo("Formation Succeed without Collision!")


if __name__ == "__main__":
    main()
