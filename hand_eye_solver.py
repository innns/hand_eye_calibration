# -*- coding: utf-8 -*-

#  Copyright (c) 2024.
# time: 24/02/26 16:31
# file: hand_eye_solver.py
# author: zx
# email: x.zhang.hz@outlook.com

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import csv
from utils import transform
import cv2

print("---------------TSAI---------------")

FILE_NAME = "hand2eye_data/hand_eye_2024-01-31-14-11.csv"
ENCODING_METHOD = "utf-8"  # UTF-8
# ENCODING_METHOD = "utf-8-sig"  # UTF-8-BOM
hands = []
eyes = []

# 打开CSV文件
with open(FILE_NAME, 'r', newline='', encoding=ENCODING_METHOD) as file:
    # 创建CSV字典读取器

    csv_reader = csv.DictReader(file)
    # 逐行读取数据
    for row in csv_reader:
        # row是一个字典，包含CSV文件中的一行数据
        hand = list([float(row['h_x']) * 1000.0, float(row['h_y']) * 1000.0, float(row['h_z']) * 1000.0,
                     float(row['h_rx']), float(row['h_ry']), float(row['h_rz'])])
        eye = list([float(row['e_x']), float(row['e_y']), float(row['e_z']),
                    float(row['e_rx']), float(row['e_ry']), float(row['e_rz'])])
        hands.append(hand)
        eyes.append(eye)
Poses_e2b = []
Poses_t2c = []
Poses_b2e = []
Poses_c2t = []
for h in hands:
    Poses_e2b.append(transform.get_transform_matrix(transform.get_r_mat_from_r_vec(np.array(h[3:])),
                                                    np.array(h[:3]).reshape(1, 3)))
    Poses_b2e.append(np.linalg.inv(Poses_e2b[-1]))

for e in eyes:
    Poses_t2c.append(transform.get_transform_matrix(transform.get_r_mat_from_r_vec(np.array(e[3:])),
                                                    np.array(e[:3]).reshape(1, 3)))
    Poses_c2t.append(np.linalg.inv(Poses_t2c[-1]))

Poses_e2b = np.array(Poses_e2b)
Poses_b2e = np.array(Poses_b2e)
Poses_t2c = np.array(Poses_t2c)
Poses_c22 = np.array(Poses_c2t)

for m in range(1):
    print("METHOD = {}".format(m))
    Trans_c2b = transform.hand_eye_calibration_eye_to_hand(Poses_b2e, Poses_t2c, m)
    print("Trans_c2b \n", Trans_c2b)
    np.save("Trans_c2b.npy", Trans_c2b)
    Trans_b2c = np.linalg.inv(Trans_c2b)
    print("Trans_b2c \n", Trans_b2c)
    np.save("Trans_b2c.npy", Trans_b2c)

    # Trans_e2t_s = []
    # Trans_t2e_s = []
    # for i in range(len(Poses_c2t)):
    #     Trans_e2t_s.append(Poses_c2t[i] @ Trans_b2c @ Poses_e2b[i])
    #     Trans_t2e_s.append(np.linalg.inv(Trans_e2t_s[-1]))
    # Trans_e2t_s = np.array(Trans_e2t_s)
    # Trans_t2e_s = np.array(Trans_t2e_s)
    min_rmse = 10000
    idx = -1
    for i in range(len(Poses_c2t)):
        origins = []
        outs = []
        Trans_t2e = np.linalg.inv(Poses_c2t[i] @ Trans_b2c @ Poses_e2b[i])
        # Trans_t2e = np.load("Trans_t2e.npy")
        for j in range(len(Poses_c2t)):
            if i != j:
                origins.append(Poses_t2c[j, :3, 3])
                # print(origin)  # tool 2 cam
                #  base 2 cam @ [end 2 base] @ tool 2 end
                # print(Trans_t2e_s[5][])
                outs.append((Trans_b2c @ Poses_e2b[j] @ Trans_t2e)[:3, 3])
        origins = np.array(origins)
        outs = np.array(outs)
        t = transform.rmse(origins, outs)
        if t < min_rmse:
            min_rmse = t
            idx = i
    print("IDX = {} RMSE = {}".format(idx, min_rmse))
    Trans_t2e = np.linalg.inv(Poses_c2t[idx] @ Trans_b2c @ Poses_e2b[idx])
    print(Trans_t2e)
    np.save("Trans_t2e.npy", Trans_t2e)  # tool to end 工具坐标系到末端坐标系的转换

    # 假设当前位置是 t 对应机械臂末端 e ； 目标位置是 t_ 对应 e_
    # Trans_c2t_ @ Trans_t2c = Trans_t2t_ = Trans_e_2t_ @ Trans_b2e_ @ Trans_e2b @ Trans_t2e
    # 其中 Trans_c2t_ 与 Trans_t2c由目标位置、当前位置决定，Trans_e_2t_ Trans_e2b Trans_t2e 都已知
    # 即可求解得到 Trans_b2e_ 可以退出机械臂需要运动到的位置
    # Trans_b2e_ = Trans_t2e @ Trans_c2t_ @ Trans_t2c @ Trans_e2t @ Trans_b2e
    # Trans_e_2b = np.linalg.inv(Trans_b2e_) 即为机械臂坐标
