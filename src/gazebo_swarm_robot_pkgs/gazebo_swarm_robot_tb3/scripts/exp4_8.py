#! /usr/bin/env python3
# encoding: utf-8

import rospy
import numpy as np
from scipy.optimize import linear_sum_assignment
from gazebo_swarm_robot_control import SwarmRobot

# ================= 策略配置 =================
# [关键修改] 选择队形B的模式
# 1: 所有机器人从中间缝隙穿过 (Staggered Column)
# 2: 机器人从障碍物两侧绕过 (Split Formation: 4 Top / 4 Bottom)
FORMATION_MODE = 2 

# ================= 通用参数 =================
ROBOT_NUM = 8
ROBOT_IDS = [i + 1 for i in range(ROBOT_NUM)]

# --- 障碍物 ---
OBSTACLE_X = 5.0        
OBS_POSITIONS = np.array([
    [OBSTACLE_X,  0.5], 
    [OBSTACLE_X, -0.5]
])
# 避障参数
OBSTACLE_SAFE_DIST = 0.30 

# --- 队形几何 ---
RADIUS_A = 0.8          
COL_SPACING_X = 0.45    

# 模式1 (中路) 的Y轴间距 (紧凑)
MODE_1_Y_SPACING = 0.24 

# 模式2 (分路) 的Y轴位置 (宽敞)
# 障碍物在 +/-0.5，我们在 +/-1.3 处走，非常安全
MODE_2_LANE_Y = 1.3     

# --- 控制参数 ---
KP = 1.0                
K_REP = 0.8             
SAFE_DIST_ROBOT = 0.25  
CONV_DIST = 0.15        
V_MOVE = 0.15           

class FormationController:
    def __init__(self):
        rospy.init_node("swarm_formation_dynamic")
        self.robot = SwarmRobot(ROBOT_IDS)
        
        self.vc_x = 0.0
        self.vc_y = 0.0
        self.state = "INIT_FORM_A" 
        self.state_start_time = rospy.Time.now()
        
        # 锁定分配结果，防止穿越时交换位置
        self.saved_assignment_indices = None
        
        rospy.sleep(1.0)
        self.update_poses()
        if len(self.poses) > 0:
            self.vc_x = np.mean(self.poses[:, 0])
            self.vc_y = np.mean(self.poses[:, 1])

        rospy.loginfo(f"Initializing with FORMATION_MODE = {FORMATION_MODE}")

    def update_poses(self):
        pose_list = self.robot.get_robot_poses()
        self.poses = np.array(pose_list) 

    def get_formation_a_points(self, cx, cy):
        """队形A：圆形"""
        points = []
        for i in range(ROBOT_NUM):
            theta = 2 * np.pi * i / ROBOT_NUM
            px = cx + RADIUS_A * np.cos(theta)
            py = cy + RADIUS_A * np.sin(theta)
            points.append([px, py])
        return np.array(points)

    def get_formation_b_points(self, cx, cy):
        """根据模式选择队形B"""
        if FORMATION_MODE == 1:
            return self._get_b_mode_center(cx, cy)
        else:
            return self._get_b_mode_split(cx, cy)

    def _get_b_mode_center(self, cx, cy):
        """模式1：中间交错纵队"""
        points = []
        rows = ROBOT_NUM // 2
        # 左列 (y略偏上)
        for i in range(rows):
            px = cx - (i * COL_SPACING_X) + (COL_SPACING_X * 1.5)
            py = cy + (MODE_1_Y_SPACING / 2.0)
            points.append([px, py])
        # 右列 (y略偏下, x交错)
        for i in range(rows):
            px = cx - (i * COL_SPACING_X) + (COL_SPACING_X * 1.5) - (COL_SPACING_X / 2.0)
            py = cy - (MODE_1_Y_SPACING / 2.0)
            points.append([px, py])
        return np.array(points)

    def _get_b_mode_split(self, cx, cy):
        """模式2：上下分路纵队 (避开中间)"""
        points = []
        half_num = ROBOT_NUM // 2 # 4个
        
        # 上路 (Top Lane): Y = +1.3
        for i in range(half_num):
            px = cx - (i * COL_SPACING_X) + (COL_SPACING_X * 1.0)
            py = cy + MODE_2_LANE_Y
            points.append([px, py])
            
        # 下路 (Bottom Lane): Y = -1.3
        for i in range(half_num):
            px = cx - (i * COL_SPACING_X) + (COL_SPACING_X * 1.0)
            py = cy - MODE_2_LANE_Y
            points.append([px, py])
            
        return np.array(points)

    def get_assigned_targets(self, raw_targets, indices):
        """使用保存的索引生成目标点"""
        sorted_targets = np.zeros_like(raw_targets)
        for i in range(ROBOT_NUM):
            target_idx = indices[i]
            sorted_targets[i] = raw_targets[target_idx]
        return sorted_targets

    def optimize_assignment(self, target_points):
        """匈牙利算法分配"""
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
        return sorted_targets, col_ind

    def calculate_repulsion(self, robot_idx):
        fx, fy = 0.0, 0.0
        cur_pos = self.poses[robot_idx, :2]

        # 1. 机器人互斥
        for i in range(ROBOT_NUM):
            if i == robot_idx: continue
            diff = cur_pos - self.poses[i, :2]
            dist = np.linalg.norm(diff)
            if dist < SAFE_DIST_ROBOT and dist > 0.01:
                rep_val = K_REP * (1.0/dist - 1.0/SAFE_DIST_ROBOT) / (dist**2)
                fx += rep_val * diff[0]
                fy += rep_val * diff[1]

        # 2. 障碍物互斥
        for obs in OBS_POSITIONS:
            diff = cur_pos - obs
            dist = np.linalg.norm(diff)
            if dist < OBSTACLE_SAFE_DIST and dist > 0.01:
                rep_val = K_REP * (1.0/dist - 1.0/OBSTACLE_SAFE_DIST) / (dist**2)
                fx += rep_val * diff[0]
                fy += rep_val * diff[1]

        # 限幅
        f_norm = np.hypot(fx, fy)
        if f_norm > 0.5: 
            fx = 0.5 * fx / f_norm
            fy = 0.5 * fy / f_norm
        return fx, fy

    def run(self):
        rate = rospy.Rate(20)
        current_targets = np.zeros((ROBOT_NUM, 2))
        
        while not rospy.is_shutdown():
            self.update_poses()
            current_time = rospy.Time.now()
            
            # --- 状态机 ---
            if self.state == "INIT_FORM_A":
                raw_targets = self.get_formation_a_points(self.vc_x, self.vc_y)
                current_targets, _ = self.optimize_assignment(raw_targets)
                
                max_err = np.max(np.linalg.norm(current_targets - self.poses[:, :2], axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("Formation A Ready. Moving Forward.")
                    self.state = "MOVE_A"

            elif self.state == "MOVE_A":
                self.vc_x += V_MOVE * 0.05
                raw_targets = self.get_formation_a_points(self.vc_x, self.vc_y)
                current_targets, _ = self.optimize_assignment(raw_targets)
                
                # 到达切换点 (3.0m 前)
                if self.vc_x > OBSTACLE_X - 3.0:
                    mode_str = "Center" if FORMATION_MODE == 1 else "Split"
                    rospy.loginfo(f"Switching to Formation B (Mode: {mode_str})...")
                    self.state = "SWITCH_TO_B"
                    self.state_start_time = rospy.Time.now()

            elif self.state == "SWITCH_TO_B":
                # 目标点稍微前移，展开队形
                target_vc_x = self.vc_x + 0.8 
                raw_targets = self.get_formation_b_points(target_vc_x, self.vc_y)
                
                # 此时使用最优分配 (让机器人选择去上面还是下面最顺路)
                current_targets, indices = self.optimize_assignment(raw_targets)
                
                max_err = np.max(np.linalg.norm(current_targets - self.poses[:, :2], axis=1))
                time_elapsed = (current_time - self.state_start_time).to_sec()
                
                # 锁定队形
                if max_err < CONV_DIST or time_elapsed > 8.0:
                    rospy.loginfo("Formation B Locked. Crossing Obstacle.")
                    self.vc_x = target_vc_x
                    # [关键] 保存分配索引
                    self.saved_assignment_indices = indices
                    self.state = "MOVE_B"

            elif self.state == "MOVE_B":
                self.vc_x += V_MOVE * 0.05
                raw_targets = self.get_formation_b_points(self.vc_x, self.vc_y)
                
                # [关键] 强制保持分配
                current_targets = self.get_assigned_targets(raw_targets, self.saved_assignment_indices)
                
                if self.vc_x > OBSTACLE_X + 2.0:
                    rospy.loginfo("Passed. Reforming A.")
                    self.state = "SWITCH_TO_A"

            elif self.state == "SWITCH_TO_A":
                target_vc_x = self.vc_x + 0.5
                raw_targets = self.get_formation_a_points(target_vc_x, self.vc_y)
                # 恢复最优分配
                current_targets, _ = self.optimize_assignment(raw_targets)
                
                max_err = np.max(np.linalg.norm(current_targets - self.poses[:, :2], axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("Mission Complete.")
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
                
                # 引力
                att_x = KP * (tx - cx)
                att_y = KP * (ty - cy)
                
                # 斥力
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