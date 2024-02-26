# hand_eye_calibration
eye to hand 眼在手外的情况下的标定程序

```shell
# activate virtual env first
# use conda or venv
# I'm using Python 3.8
python3 -m pip install numpy scipy matplotlib opencv-python
```

h_x, h_y, h_z, h_rx, h_ry, h_rz 分别代表着机械臂读取到的末端坐标与旋转向量
e_x, e_y, e_z, e_rx, e_ry, e_rz 分别代表着相机/定位系统读取到的器械/棋盘格坐标与旋转向量

注意代码中需要 统一到 `mm` `rad` 为单位
