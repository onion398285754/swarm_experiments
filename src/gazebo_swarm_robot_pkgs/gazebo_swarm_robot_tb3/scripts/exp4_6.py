#! /usr/bin/env python3
# encoding: utf-8

import rospy
import numpy as np
from scipy.optimize import linear_sum_assignment
from gazebo_swarm_robot_control import SwarmRobot

# ================= 策略配置 =================
# 1: 所有机器人从中间缝隙穿过 (Staggered Column)
# 2: 机器人从障碍物两侧绕过 (Split Formation: 3 Top / 3 Bottom)
FORMATION_MODE = 2 

# ================= 通用参数 =================
ROBOT_NUM = 6
ROBOT_IDS = [i + 1 for i in range(ROBOT_NUM)]

# --- 时间控制 (单位：秒) ---
TIME_MOVE_A = 8.0         # 第一阶段圆阵行走时间
TIME_MOVE_B = 10.0        # 第二阶段避障队形行走时间
TIME_MOVE_A_FINAL = 6.0   # 第三阶段恢复圆阵后行走时间

# --- 避障与安全 ---
SAFE_DIST_ROBOT = 0.35    # 机器人间安全距离
K_REP = 1.2               # 斥力系数
CONV_DIST = 0.15          # 判定队形形成的精度

# --- 队形几何 ---
RADIUS_A = 1.0            # 圆阵半径
COL_SPACING_X = 0.60      # 纵队前后间距
MODE_1_Y_SPACING = 0.45   # 模式1：中路纵队左右间距
MODE_2_LANE_Y = 1.3       # 模式2：分路行走Y轴位置

# --- 控制参数 ---
V_MOVE = 0.2         
KP = 1.0             
OBSTACLE_SAFE_DIST = 0.4
OBS_POSITIONS = np.array([[5.0, 0.5], [5.0, -0.5]])

class FormationController:
    def __init__(self):
        rospy.init_node("swarm_formation_timer_fixed")
        self.robot = SwarmRobot(ROBOT_IDS)
        
        self.vc_x = 0.0
        self.vc_y = 0.0
        # 状态机：原地变阵A -> 行走A -> 原地变阵B -> 行走B -> 原地恢复A -> 行走A
        self.state = "FORM_A"
        self.state_start_time = 0
        self.saved_indices = None
        
        rospy.sleep(1.0)
        self.update_poses()
        if len(self.poses) > 0:
            self.vc_x = np.mean(self.poses[:, 0])
            self.vc_y = np.mean(self.poses[:, 1])
        
        rospy.loginfo(f"Start with MODE {FORMATION_MODE}, Robots: {ROBOT_NUM}")

    def update_poses(self):
        pose_list = self.robot.get_robot_poses()
        self.poses = np.array(pose_list) 

    def get_formation_a_points(self, cx, cy):
        points = []
        for i in range(ROBOT_NUM):
            theta = 2 * np.pi * i / ROBOT_NUM
            points.append([cx + RADIUS_A * np.cos(theta), cy + RADIUS_A * np.sin(theta)])
        return np.array(points)

    def get_formation_b_points(self, cx, cy):
        if FORMATION_MODE == 1:
            return self._get_b_mode_center(cx, cy)
        else:
            return self._get_b_mode_split(cx, cy)

    def _get_b_mode_center(self, cx, cy):
        """模式1：中间双排纵队"""
        points = []
        rows = ROBOT_NUM // 2
        x_off = (rows - 1) * COL_SPACING_X / 2.0
        for i in range(rows): # 左列
            points.append([cx + x_off - i * COL_SPACING_X, cy + MODE_1_Y_SPACING/2.0])
        for i in range(rows): # 右列 (交错)
            points.append([cx + x_off - i * COL_SPACING_X - 0.2, cy - MODE_1_Y_SPACING/2.0])
        return np.array(points)

    def _get_b_mode_split(self, cx, cy):
        """模式2：上下分路"""
        points = []
        half = ROBOT_NUM // 2
        x_off = (half - 1) * COL_SPACING_X / 2.0
        for i in range(half): # 上路
            points.append([cx + x_off - i * COL_SPACING_X, cy + MODE_2_LANE_Y])
        for i in range(half): # 下路
            points.append([cx + x_off - i * COL_SPACING_X, cy - MODE_2_LANE_Y])
        return np.array(points)

    def get_targets(self, raw_points):
        """使用匈牙利算法动态分配最优目标点"""
        cost = np.zeros((ROBOT_NUM, ROBOT_NUM))
        for i in range(ROBOT_NUM): 
            for j in range(ROBOT_NUM): 
                cost[i, j] = np.linalg.norm(self.poses[i, :2] - raw_points[j])
        _, col_ind = linear_sum_assignment(cost)
        return raw_points[col_ind], col_ind

    def calculate_repulsion(self, idx):
        fx, fy = 0.0, 0.0
        cp = self.poses[idx, :2]
        # 机器人间斥力
        for i in range(ROBOT_NUM):
            if i == idx: continue
            diff = cp - self.poses[i, :2]
            d = np.linalg.norm(diff)
            if 0.01 < d < SAFE_DIST_ROBOT:
                rep = K_REP * (1.0/d - 1.0/SAFE_DIST_ROBOT) / (d**2)
                fx += rep * diff[0] / d
                fy += rep * diff[1] / d
        # 障碍物斥力
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
        
        while not rospy.is_shutdown():
            self.update_poses()
            now = rospy.Time.now().to_sec()
            
            # --- 状态机逻辑 ---
            if self.state == "FORM_A":
                # 原地形成圆阵
                raw = self.get_formation_a_points(self.vc_x, self.vc_y)
                targets, _ = self.get_targets(raw)
                max_err = np.max(np.linalg.norm(self.poses[:, :2] - targets, axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("FORM_A Done. Walking...")
                    self.state = "MOVE_A"
                    self.state_start_time = now

            elif self.state == "MOVE_A":
                # 圆阵计时行走
                self.vc_x += V_MOVE * 0.05
                raw = self.get_formation_a_points(self.vc_x, self.vc_y)
                targets, _ = self.get_targets(raw)
                if now - self.state_start_time > TIME_MOVE_A:
                    rospy.loginfo("MOVE_A Time Up. Forming B...")
                    self.state = "FORM_B"
                    self.saved_indices = None

            elif self.state == "FORM_B":
                # 原地变换为避障队形 (MODE 1 或 2)
                raw = self.get_formation_b_points(self.vc_x, self.vc_y)
                targets, current_idx = self.get_targets(raw)
                max_err = np.max(np.linalg.norm(self.poses[:, :2] - targets, axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("FORM_B Done. Walking through obstacles...")
                    self.state = "MOVE_B"
                    self.state_start_time = now
                    self.saved_indices = current_idx # 锁定分配

            elif self.state == "MOVE_B":
                # 避障队形计时行走
                self.vc_x += V_MOVE * 0.05
                raw = self.get_formation_b_points(self.vc_x, self.vc_y)
                targets = raw[self.saved_indices] # 保持锁定防止碰撞
                if now - self.state_start_time > TIME_MOVE_B:
                    rospy.loginfo("MOVE_B Time Up. Returning to A...")
                    self.state = "FORM_A_FINAL"
                    self.saved_indices = None

            elif self.state == "FORM_A_FINAL":
                # 原地恢复圆阵
                raw = self.get_formation_a_points(self.vc_x, self.vc_y)
                targets, _ = self.get_targets(raw)
                max_err = np.max(np.linalg.norm(self.poses[:, :2] - targets, axis=1))
                if max_err < CONV_DIST:
                    rospy.loginfo("FORM_A_FINAL Done. Final stretch...")
                    self.state = "MOVE_A_FINAL"
                    self.state_start_time = now

            elif self.state == "MOVE_A_FINAL":
                # 最后圆阵行走
                self.vc_x += V_MOVE * 0.05
                raw = self.get_formation_a_points(self.vc_x, self.vc_y)
                targets, _ = self.get_targets(raw)
                if now - self.state_start_time > TIME_MOVE_A_FINAL:
                    self.state = "DONE"

            elif self.state == "DONE":
                self.robot.stop_robots()
                rospy.loginfo("Mission Complete.")
                break

            # --- 控制指令生成 ---
            u_x, u_y = [], []
            for i in range(ROBOT_NUM):
                rx, ry = self.calculate_repulsion(i)
                # 斥力限幅，保证平稳
                f_rep = np.hypot(rx, ry)
                if f_rep > 0.8:
                    rx, ry = 0.8 * rx/f_rep, 0.8 * ry/f_rep
                
                u_x.append(KP * (targets[i, 0] - self.poses[i, 0]) + rx)
                u_y.append(KP * (targets[i, 1] - self.poses[i, 1]) + ry)
            
            self.robot.move_robots_by_u(u_x, u_y)
            rate.sleep()

if __name__ == "__main__":
    try:
        FormationController().run()
    except rospy.ROSInterruptException:
        pass