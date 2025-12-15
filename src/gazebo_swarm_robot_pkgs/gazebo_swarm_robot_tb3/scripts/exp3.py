#! /usr/bin/env python3
# encoding: utf-8

import rospy
import numpy as np
from scipy.optimize import linear_sum_assignment
from gazebo_swarm_robot_control import SwarmRobot

# ================= 配置参数 =================
ROBOT_NUM = 8
ROBOT_IDS = [i + 1 for i in range(ROBOT_NUM)]

# --- 障碍物 ---
OBSTACLE_X = 5.0        
OBS_POSITIONS = np.array([
    [OBSTACLE_X,  0.5], 
    [OBSTACLE_X, -0.5]
])
# [修改1] 减小障碍物避障半径，防止机器人"蹭"到斥力场
# 之前是0.35，改为0.30，给机器人留出 0.5-0.3=0.2m 的绝对安全区
OBSTACLE_SAFE_DIST = 0.30 

# --- 队形 ---
RADIUS_A = 0.8          
COL_SPACING_X = 0.45    
# [修改2] 稍微收窄纵队宽度，让机器人离柱子更远
# 之前是0.30，改为0.24 (即y=±0.12)，这样离障碍物(y=0.5)有0.38m距离 > 0.30斥力区
COL_SPACING_Y = 0.24    

# --- 控制 ---
KP = 1.0                
K_REP = 0.8             # 稍微恢复一点斥力，保证不互撞
SAFE_DIST_ROBOT = 0.25  
CONV_DIST = 0.15        
V_MOVE = 0.15           

class FormationController:
    def __init__(self):
        rospy.init_node("swarm_formation_stable")
        self.robot = SwarmRobot(ROBOT_IDS)
        
        self.vc_x = 0.0
        self.vc_y = 0.0
        self.state = "INIT_FORM_A" 
        self.state_start_time = rospy.Time.now()
        
        # [新增] 用于存储分配结果，锁定队形
        self.saved_assignment_indices = None
        
        rospy.sleep(1.0)
        self.update_poses()
        if len(self.poses) > 0:
            self.vc_x = np.mean(self.poses[:, 0])
            self.vc_y = np.mean(self.poses[:, 1])

    def update_poses(self):
        pose_list = self.robot.get_robot_poses()
        self.poses = np.array(pose_list) 

    def get_formation_a_points(self, center_x, center_y):
        points = []
        for i in range(ROBOT_NUM):
            theta = 2 * np.pi * i / ROBOT_NUM
            px = center_x + RADIUS_A * np.cos(theta)
            py = center_y + RADIUS_A * np.sin(theta)
            points.append([px, py])
        return np.array(points)

    def get_formation_b_points(self, center_x, center_y):
        """交错式双排"""
        points = []
        rows = ROBOT_NUM // 2
        # 左列
        for i in range(rows):
            px = center_x - (i * COL_SPACING_X) + (COL_SPACING_X * 1.5)
            py = center_y + (COL_SPACING_Y / 2.0)
            points.append([px, py])
        # 右列 (交错)
        for i in range(rows):
            px = center_x - (i * COL_SPACING_X) + (COL_SPACING_X * 1.5) - (COL_SPACING_X / 2.0)
            py = center_y - (COL_SPACING_Y / 2.0)
            points.append([px, py])
        return np.array(points)

    def get_assigned_targets(self, raw_targets, indices):
        """
        [新增] 根据保存的索引直接生成目标点，不再计算最小距离
        indices[i] 表示第 i 个机器人应该去 raw_targets 中的哪个点
        """
        sorted_targets = np.zeros_like(raw_targets)
        for i in range(ROBOT_NUM):
            target_idx = indices[i]
            sorted_targets[i] = raw_targets[target_idx]
        return sorted_targets

    def optimize_assignment(self, target_points):
        """计算最优分配并返回 排序后的点 和 分配索引"""
        cost_matrix = np.zeros((ROBOT_NUM, ROBOT_NUM))
        for i in range(ROBOT_NUM): 
            for j in range(ROBOT_NUM): 
                dist_sq = (self.poses[i, 0] - target_points[j, 0])**2 + \
                          (self.poses[i, 1] - target_points[j, 1])**2
                cost_matrix[i, j] = dist_sq
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        sorted_targets = np.zeros_like(target_points)
        for i in range(ROBOT_NUM):
            sorted_targets[i] = target_points[col_ind[i]]
            
        return sorted_targets, col_ind # 返回 col_ind 以便保存

    def calculate_repulsion(self, robot_idx):
        fx, fy = 0.0, 0.0
        cur_pos = self.poses[robot_idx, :2]

        for i in range(ROBOT_NUM):
            if i == robot_idx: continue
            diff = cur_pos - self.poses[i, :2]
            dist = np.linalg.norm(diff)
            if dist < SAFE_DIST_ROBOT and dist > 0.01:
                rep_val = K_REP * (1.0/dist - 1.0/SAFE_DIST_ROBOT) / (dist**2)
                fx += rep_val * diff[0]
                fy += rep_val * diff[1]

        for obs in OBS_POSITIONS:
            diff = cur_pos - obs
            dist = np.linalg.norm(diff)
            # 斥力触发区域检查
            if dist < OBSTACLE_SAFE_DIST and dist > 0.01:
                rep_val = K_REP * (1.0/dist - 1.0/OBSTACLE_SAFE_DIST) / (dist**2)
                fx += rep_val * diff[0]
                fy += rep_val * diff[1]

        f_norm = np.hypot(fx, fy)
        if f_norm > 0.5: 
            fx = 0.5 * fx / f_norm
            fy = 0.5 * fy / f_norm
        return fx, fy

    def run(self):
        rate = rospy.Rate(20)
        current_targets = np.zeros((ROBOT_NUM, 2))
        rospy.loginfo("Start Mission: Stable Crossing...")
        
        while not rospy.is_shutdown():
            self.update_poses()
            current_time = rospy.Time.now()
            
            # --- 状态机 ---
            if self.state == "INIT_FORM_A":
                raw_targets = self.get_formation_a_points(self.vc_x, self.vc_y)
                # 初始阶段持续优化分配
                current_targets, _ = self.optimize_assignment(raw_targets)
                
                max_err = np.max(np.linalg.norm(current_targets - self.poses[:, :2], axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("Formation A Ready.")
                    self.state = "MOVE_A"

            elif self.state == "MOVE_A":
                self.vc_x += V_MOVE * 0.05
                raw_targets = self.get_formation_a_points(self.vc_x, self.vc_y)
                current_targets, _ = self.optimize_assignment(raw_targets)
                
                if self.vc_x > OBSTACLE_X - 3.0:
                    rospy.loginfo("Switching to Formation B...")
                    self.state = "SWITCH_TO_B"
                    self.state_start_time = rospy.Time.now()

            elif self.state == "SWITCH_TO_B":
                target_vc_x = self.vc_x + 0.8 
                raw_targets = self.get_formation_b_points(target_vc_x, self.vc_y)
                
                # 在变换过程中，我们持续优化，找到谁去哪里最合适
                current_targets, indices = self.optimize_assignment(raw_targets)
                
                max_err = np.max(np.linalg.norm(current_targets - self.poses[:, :2], axis=1))
                time_elapsed = (current_time - self.state_start_time).to_sec()
                
                if max_err < CONV_DIST or time_elapsed > 8.0:
                    rospy.loginfo("Formation B Locked. Moving Forward.")
                    self.vc_x = target_vc_x
                    
                    # [关键修改] 锁定当前的分配方案！
                    # 记录下"哪个机器人去了哪个位置"，之后不再改变
                    self.saved_assignment_indices = indices
                    
                    self.state = "MOVE_B"

            elif self.state == "MOVE_B":
                self.vc_x += V_MOVE * 0.05
                raw_targets = self.get_formation_b_points(self.vc_x, self.vc_y)
                
                # [关键修改] 不再调用 optimize_assignment
                # 而是使用 SWITCH_TO_B 结束时锁定的 indices
                current_targets = self.get_assigned_targets(raw_targets, self.saved_assignment_indices)
                
                if self.vc_x > OBSTACLE_X + 2.0:
                    rospy.loginfo("Passed. Reforming A.")
                    self.state = "SWITCH_TO_A"

            elif self.state == "SWITCH_TO_A":
                target_vc_x = self.vc_x + 0.5
                raw_targets = self.get_formation_a_points(target_vc_x, self.vc_y)
                # 恢复优化分配，以便变回圆形时能顺滑解散
                current_targets, _ = self.optimize_assignment(raw_targets)
                
                max_err = np.max(np.linalg.norm(current_targets - self.poses[:, :2], axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("Done.")
                    self.vc_x = target_vc_x
                    self.state = "DONE"
            
            elif self.state == "DONE":
                self.robot.stop_robots()
                break

            # --- 控制输出 ---
            u_x_list, u_y_list = [], []
            for i in range(ROBOT_NUM):
                tx, ty = current_targets[i]
                cx, cy = self.poses[i, 0], self.poses[i, 1]
                
                att_x = KP * (tx - cx)
                att_y = KP * (ty - cy)
                
                rep_x, rep_y = self.calculate_repulsion(i)
                
                u_x_list.append(att_x + rep_x)
                u_y_list.append(att_y + rep_y)
            
            self.robot.move_robots_by_u(u_x_list, u_y_list)
            rate.sleep()

if __name__ == "__main__":
    try:
        controller = FormationController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
