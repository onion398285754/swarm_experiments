#! /usr/bin/env python3
# encoding: utf-8

import rospy
import numpy as np
from scipy.optimize import linear_sum_assignment
from gazebo_swarm_robot_control import SwarmRobot

# ================= 配置参数 =================
ROBOT_NUM = 6
ROBOT_IDS = [i + 1 for i in range(ROBOT_NUM)]

# --- 时间控制 ---
TIME_MOVE_A = 8.0         # 第一阶段队形A行走时间
TIME_MOVE_B = 8.0         # 第二阶段队形B行走时间
TIME_MOVE_A_FINAL = 6.0   # [新增] 第三阶段恢复队形A后的行走时间

# --- 避障与安全 ---
SAFE_DIST_ROBOT = 0.35  
K_REP = 1.2             
CONV_DIST = 0.15        

# --- 队形尺寸 ---
RADIUS_A = 1.0          
COL_SPACING_X = 0.60    
COL_SPACING_Y = 0.45    

# --- 控制 ---
V_MOVE = 0.2         
KP = 1.0             
OBSTACLE_SAFE_DIST = 0.4
OBS_POSITIONS = np.array([[5.0, 0.5], [5.0, -0.5]])

class FormationController:
    def __init__(self):
        rospy.init_node("swarm_safe_formation_loop")
        self.robot = SwarmRobot(ROBOT_IDS)
        
        self.vc_x = 0.0
        self.vc_y = 0.0
        # 状态机：FORM_A -> MOVE_A -> FORM_B -> MOVE_B -> FORM_A_FINAL -> MOVE_A_FINAL -> DONE
        self.state = "FORM_A"
        self.state_start_time = 0
        self.saved_indices = None
        
        rospy.sleep(1.0)
        self.update_poses()
        if len(self.poses) > 0:
            self.vc_x = np.mean(self.poses[:, 0])
            self.vc_y = np.mean(self.poses[:, 1])

    def update_poses(self):
        pose_list = self.robot.get_robot_poses()
        self.poses = np.array(pose_list)

    def get_formation_a(self, cx, cy):
        points = []
        for i in range(ROBOT_NUM):
            theta = 2 * np.pi * i / ROBOT_NUM
            points.append([cx + RADIUS_A * np.cos(theta), cy + RADIUS_A * np.sin(theta)])
        return np.array(points)

    def get_formation_b(self, cx, cy):
        points = []
        rows = ROBOT_NUM // 2
        x_off = (rows - 1) * COL_SPACING_X / 2.0
        for i in range(rows): # 左列
            points.append([cx + x_off - i * COL_SPACING_X, cy + COL_SPACING_Y/2.0])
        for i in range(rows): # 右列
            points.append([cx + x_off - i * COL_SPACING_X, cy - COL_SPACING_Y/2.0])
        return np.array(points)

    def get_targets(self, raw_points):
        cost = np.zeros((ROBOT_NUM, ROBOT_NUM))
        for i in range(ROBOT_NUM):
            for j in range(ROBOT_NUM):
                dist = np.linalg.norm(self.poses[i,:2] - raw_points[j])
                cost[i,j] = dist
        _, col_ind = linear_sum_assignment(cost)
        return raw_points[col_ind], col_ind

    def calculate_repulsion(self, idx):
        fx, fy = 0.0, 0.0
        cp = self.poses[idx, :2]
        for i in range(ROBOT_NUM):
            if i == idx: continue
            diff = cp - self.poses[i, :2]
            d = np.linalg.norm(diff)
            if 0.01 < d < SAFE_DIST_ROBOT:
                rep = K_REP * (1.0/d - 1.0/SAFE_DIST_ROBOT) / (d**2)
                fx += rep * diff[0] / d
                fy += rep * diff[1] / d
        for obs in OBS_POSITIONS:
            diff = cp - obs
            d = np.linalg.norm(diff)
            if 0.01 < d < OBSTACLE_SAFE_DIST:
                rep = K_REP * (1.0/d - 1.0/OBSTACLE_SAFE_DIST) / (d**2)
                fx += rep * diff[0] / d
                fy += rep * diff[1] / d
        return fx, fy

    def run(self):
        rate = rospy.Rate(20)
        rospy.loginfo("Mission Started.")

        while not rospy.is_shutdown():
            self.update_poses()
            now = rospy.Time.now().to_sec()

            if self.state == "FORM_A":
                raw = self.get_formation_a(self.vc_x, self.vc_y)
                targets, _ = self.get_targets(raw)
                max_err = np.max(np.linalg.norm(self.poses[:,:2] - targets, axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("Formed A. Walking...")
                    self.state = "MOVE_A"
                    self.state_start_time = now

            elif self.state == "MOVE_A":
                self.vc_x += V_MOVE * 0.05
                raw = self.get_formation_a(self.vc_x, self.vc_y)
                targets, _ = self.get_targets(raw)
                if now - self.state_start_time > TIME_MOVE_A:
                    rospy.loginfo("Time up A. Forming B...")
                    self.state = "FORM_B"
                    self.saved_indices = None 

            elif self.state == "FORM_B":
                raw = self.get_formation_b(self.vc_x, self.vc_y)
                targets, current_indices = self.get_targets(raw)
                max_err = np.max(np.linalg.norm(self.poses[:,:2] - targets, axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("Formed B. Walking through...")
                    self.state = "MOVE_B"
                    self.state_start_time = now
                    self.saved_indices = current_indices

            elif self.state == "MOVE_B":
                self.vc_x += V_MOVE * 0.05
                raw = self.get_formation_b(self.vc_x, self.vc_y)
                targets = raw[self.saved_indices]
                if now - self.state_start_time > TIME_MOVE_B:
                    # [关键修改] 修改这里，不再直接DONE，而是切换回FORM_A
                    rospy.loginfo("Time up B. Returning to A...")
                    self.state = "FORM_A_FINAL"
                    self.saved_indices = None

            elif self.state == "FORM_A_FINAL":
                # [新增] 原地重新形成圆阵
                raw = self.get_formation_a(self.vc_x, self.vc_y)
                targets, _ = self.get_targets(raw)
                max_err = np.max(np.linalg.norm(self.poses[:,:2] - targets, axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("Formed A again. Walking final stretch...")
                    self.state = "MOVE_A_FINAL"
                    self.state_start_time = now

            elif self.state == "MOVE_A_FINAL":
                # [新增] 最后的圆阵行走
                self.vc_x += V_MOVE * 0.05
                raw = self.get_formation_a(self.vc_x, self.vc_y)
                targets, _ = self.get_targets(raw)
                if now - self.state_start_time > TIME_MOVE_A_FINAL:
                    rospy.loginfo("Mission Complete.")
                    self.state = "DONE"

            elif self.state == "DONE":
                self.robot.stop_robots()
                break

            # 执行控制
            ux, uy = [], []
            for i in range(ROBOT_NUM):
                rx, ry = self.calculate_repulsion(i)
                f_rep = np.hypot(rx, ry)
                if f_rep > 0.8:
                    rx, ry = 0.8 * rx/f_rep, 0.8 * ry/f_rep
                
                ux.append(KP * (targets[i,0] - self.poses[i,0]) + rx)
                uy.append(KP * (targets[i,1] - self.poses[i,1]) + ry)
            
            self.robot.move_robots_by_u(ux, uy)
            rate.sleep()

if __name__ == "__main__":
    try:
        FormationController().run()
    except rospy.ROSInterruptException: pass