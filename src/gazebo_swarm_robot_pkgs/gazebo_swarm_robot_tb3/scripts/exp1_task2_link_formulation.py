#! /usr/bin/env python3
# encoding: utf-8
import numpy as np
import rospy
from gazebo_swarm_robot_control import SwarmRobot


def main():
    rospy.init_node("swarm_robot_line_formation_stationary")
    robot_ids = [1, 2, 3, 4, 5]
    swarm_robot = SwarmRobot(robot_ids)
    num_robots = len(robot_ids)

    # --- 控制参数 ---
    # "引力"增益：机器人向其在线上的目标点移动的速度
    k_formation = 0.4
    # "斥力"增益：机器人互相排斥的强度
    k_repulsion = 0.3
    # 安全距离：小于此距离将触发斥力
    # 这个值需要根据你的机器人模型大小来调整，TurtleBot3 大约是 0.4-0.5 米
    safe_distance = 0.5
    
    # --- 收敛判断参数 ---
    # 所有机器人离其目标点的总距离小于此值，则认为队形收敛
    conv_dist_total_th = 0.15 
    # 所有机器人的期望速度大小之和小于此值，则认为机器人已趋于静止
    conv_speed_total_th = 0.05
    
    rospy.loginfo("Starting stationary line formation control...")

    # 循环运行，直到队形收敛且机器人基本静止
    while not rospy.is_shutdown():
        # 1.获取所有机器人的位姿
        # --- 主要修改点 ---
        # 直接用一个变量接收 get_robot_poses() 的返回值（即位姿列表）
        poses = swarm_robot.get_robot_poses()
        
        # 由于原始的 get_robot_poses() 会一直阻塞直到成功，
        # 所以不再需要 success 变量和相关的错误检查。
        # (已删除 'if not success:' 检查块)
            
        positions = np.array([[p[0], p[1]] for p in poses])

        # 2. 定义目标直线 (由机器人1和5的位置决定)
        p_start = positions[0]
        p_end = positions[-1]
        line_vec = p_end - p_start

        # 3. 计算每个机器人的总期望速度 (ux, uy)
        ux = np.zeros(num_robots)
        uy = np.zeros(num_robots)
        total_distance_error = 0.0

        for i in range(num_robots):
            p_current = positions[i]

            # --- a. 计算“引力”：朝向线上目标点的速度 ---
            # 计算机器人 i 在直线上的目标位置 (等间距排列)
            # 对于首尾机器人，其目标点就是当前点，所以引力为0
            target_pos = p_start + (i / (num_robots - 1)) * line_vec
            
            # 计算从当前位置指向目标位置的向量
            vec_to_target = target_pos - p_current
            u_formation = k_formation * vec_to_target
            
            # 累加总的位置误差，用于判断收敛
            total_distance_error += np.linalg.norm(vec_to_target)

            # --- b. 计算“斥力”：来自其他机器人的排斥速度 ---
            u_repulsion = np.array([0.0, 0.0])
            for j in range(num_robots):
                if i == j:
                    continue

                p_neighbor = positions[j]
                dist = np.linalg.norm(p_current - p_neighbor)

                if dist < safe_distance:
                    # 计算排斥力方向 (从邻居指向自己)
                    vec_away_from_neighbor = p_current - p_neighbor
                    # 斥力大小与靠近程度成反比
                    magnitude = k_repulsion * (safe_distance - dist) / dist
                    u_repulsion += magnitude * vec_away_from_neighbor

            # --- c. 合成最终速度 ---
            # 总速度 = 引力 + 斥力
            u_total = u_formation + u_repulsion
            ux[i] = u_total[0]
            uy[i] = u_total[1]

        # 4. 应用计算出的速度
        swarm_robot.move_robots_by_u(ux, uy)

        # 5. 检查是否收敛
        # 计算所有机器人期望速度的大小之和
        total_speed_magnitude = np.sum(np.sqrt(np.square(ux) + np.square(uy)))
        
        rospy.loginfo_throttle(1, f"Total distance error: {total_distance_error:.3f}, Total speed: {total_speed_magnitude:.3f}")

        if total_distance_error < conv_dist_total_th and total_speed_magnitude < conv_speed_total_th:
            rospy.loginfo("Line formation achieved and robots are stable!")
            break

        rospy.sleep(0.1)

    # 任务完成，停止所有机器人
    swarm_robot.stop_robots()
    rospy.loginfo("Task finished.")


if __name__ == "__main__":
    main()