#!/usr/bin/env python3
# encoding: utf-8
"""
Date: 2024-12-11
Description: 控制机器人集群从任意初始位姿，通过动态角色分配和无碰撞路径规划，
             构建楔形编队。
"""

import rospy
import numpy as np
import time
# 导入您提供的基础控制类
from gazebo_swarm_robot_control import SwarmRobot
# 导入用于解决分配问题的匈牙利算法
from scipy.optimize import linear_sum_assignment

class IntelligentFormation(SwarmRobot):
    """
    实现智能编队控制，包含动态分配和碰撞避免。
    """

    def __init__(self, swarm_robot_id: list, d_safe=0.6, k_att=0.7, k_rep=0.9):
        """
        初始化智能编队控制器。

        Args:
            swarm_robot_id (list): 机器人ID列表。
            d_safe (float): 触发碰撞避免的安全距离 (米)。
            k_att (float): 目标引力的增益系数。
            k_rep (float): 机器人间斥力的增益系数。
        """
        super().__init__(swarm_robot_id)
        self.formation_center = [0.0, 0.0]  # 队形中心点坐标
        self.D_SAFE = d_safe
        self.K_ATT = k_att
        self.K_REP = k_rep
        rospy.loginfo(f"智能编队控制器已初始化。安全距离: {d_safe}m。")

    def get_wedge_positions(self, width=2.5, height=2.0):
        """
        计算楔形编队中每个机器人的目标位置点。

        Args:
            width (float): 楔形队形的宽度。
            height (float): 楔形队形的高度。

        Returns:
            list: 一个包含每个机器人目标 [x, y] 坐标的列表。
        """
        target_slots = []
        num_bots = self.robot_num

        if num_bots < 3:
            rospy.logerr("楔形队形至少需要3个机器人。")
            return None
        
        # 领队机器人的位置
        leader_pos = [self.formation_center[0], self.formation_center[1] + height / 2.0]
        target_slots.append(leader_pos)
        
        # 两翼机器人的位置
        num_wing_bots = num_bots - 1
        # 确保两边尽可能平均分配
        left_wing_num = (num_wing_bots + 1) // 2
        right_wing_num = num_wing_bots - left_wing_num
        
        # 左翼 (从领队身后开始)
        for i in range(1, left_wing_num + 1):
            target_slots.append([
                leader_pos[0] - i * (width / 2.0) / left_wing_num,
                leader_pos[1] - i * height / left_wing_num,
            ])

        # 右翼 (从领队身后开始)
        for i in range(1, right_wing_num + 1):
            target_slots.append([
                leader_pos[0] + i * (width / 2.0) / right_wing_num,
                leader_pos[1] - i * height / right_wing_num,
            ])
            
        return target_slots

    def assign_targets_dynamically(self, current_poses, target_slots):
        """
        动态地为每个机器人分配最优的目标点，以最小化总移动距离。

        Args:
            current_poses (list): 所有机器人当前的 [x, y, yaw] 位姿列表。
            target_slots (list): 所有可用的目标 [x, y] 位置列表。

        Returns:
            list: 一个重新排序后的目标位置列表，其索引与机器人索引对应。
        """
        num_bots = len(current_poses)
        cost_matrix = np.zeros((num_bots, num_bots))

        # 1. 构建成本矩阵：cost_matrix[i][j] = 机器人i到目标点j的距离
        for i in range(num_bots):
            for j in range(num_bots):
                pos_robot = current_poses[i]
                pos_target = target_slots[j]
                distance = np.sqrt((pos_robot[0] - pos_target[0])**2 + (pos_robot[1] - pos_target[1])**2)
                cost_matrix[i, j] = distance

        # 2. 使用匈牙利算法求解最优分配
        robot_indices, target_indices = linear_sum_assignment(cost_matrix)

        # 3. 根据分配结果创建新的目标列表
        assigned_targets = [None] * num_bots
        for i, robot_idx in enumerate(robot_indices):
            target_idx = target_indices[i]
            assigned_targets[robot_idx] = target_slots[target_idx]
        
        rospy.loginfo("动态目标分配完成。")
        return assigned_targets

    def form_wedge_formation(self, width=2.5, height=2.0, conv_threshold=0.15, timeout=120):
        """
        控制机器人集群形成楔形编队，具有动态分配和防撞功能。
        """
        # 1. 获取队形的目标“槽位”
        target_slots = self.get_wedge_positions(width, height)
        if not target_slots:
            return
        rospy.loginfo(f"计算楔形编队目标点: {target_slots}")

        # 2. 获取机器人当前位置
        initial_poses = self.get_robot_poses()

        # 3. 动态分配最优目标点
        assigned_targets = self.assign_targets_dynamically(initial_poses, target_slots)

        # 4. 主控制循环
        start_time = time.time()
        rospy.loginfo("开始移动至目标队形 (带碰撞躲避)...")
        
        while not rospy.is_shutdown() and time.time() - start_time < timeout:
            current_poses = self.get_robot_poses()
            
            ux_total = [0.0] * self.robot_num
            uy_total = [0.0] * self.robot_num
            
            all_robots_converged = True

            for i in range(self.robot_num):
                # --- A. 计算引力 (Attractive Force) ---
                target_pos = assigned_targets[i]
                current_pos = current_poses[i]
                
                error_x = target_pos[0] - current_pos[0]
                error_y = target_pos[1] - current_pos[1]
                distance_to_target = np.sqrt(error_x**2 + error_y**2)

                if distance_to_target > conv_threshold:
                    all_robots_converged = False
                    # 引力与到目标的距离成正比
                    ux_att = self.K_ATT * error_x
                    uy_att = self.K_ATT * error_y
                else:
                    # 到达目标附近，引力为0
                    ux_att, uy_att = 0.0, 0.0

                # --- B. 计算斥力 (Repulsive Force) ---
                ux_rep, uy_rep = 0.0, 0.0
                for j in range(self.robot_num):
                    if i == j:
                        continue
                    
                    other_pos = current_poses[j]
                    dist_vec = np.array(current_pos[:2]) - np.array(other_pos[:2])
                    distance_ij = np.linalg.norm(dist_vec)

                    # 如果距离小于安全距离，则计算斥力
                    if distance_ij < self.D_SAFE and distance_ij > 0.01:
                        # 斥力大小与距离成反比
                        force_mag = self.K_REP * (1.0/distance_ij - 1.0/self.D_SAFE) * (1.0 / (distance_ij**2))
                        ux_rep += force_mag * (dist_vec[0] / distance_ij)
                        uy_rep += force_mag * (dist_vec[1] / distance_ij)
                
                # --- C. 合成总的控制指令 ---
                ux_total[i] = ux_att + ux_rep
                uy_total[i] = uy_att + uy_rep

            # 如果所有机器人都到达了目标位置，则任务完成
            if all_robots_converged:
                rospy.loginfo("所有机器人均到达目标位置，队形构建完成!")
                self.stop_robots()
                return
            
            # 根据计算出的(ux, uy)移动所有机器人
            self.move_robots_by_u(ux_total, uy_total)
            rospy.sleep(0.1)

        rospy.logwarn("在规定时间内未能完成队形构建。")
        self.stop_robots()

def main():
    # 初始化ROS节点
    rospy.init_node("intelligent_formation_control")

    try:
        # 定义机器人ID列表
        robot_ids = [1, 2, 3, 4, 5]

        # 创建控制器实例，可以调整这里的参数
        # d_safe: 机器人开始躲避的距离
        # k_att: 飞向目标的“力度”
        # k_rep: 躲避其他机器人的“力度”
        controller = IntelligentFormation(
            swarm_robot_id=robot_ids,
            d_safe=0.5, 
            k_att=0.8,
            k_rep=1.0
        )
        
        # 等待所有系统（如TF）准备就绪
        rospy.sleep(1)

        # --- 执行唯一的任务: 构建楔形编队 ---
        rospy.loginfo("********** 开始构建楔形编队 **********")
        controller.form_wedge_formation(width=3.0, height=2.5)
        
        rospy.loginfo("任务完成。")

    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()