import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import orth
from scipy.optimize import least_squares, LinearConstraint, NonlinearConstraint
from scipy.linalg import logm, expm
import cv2


def kabsch(p, q):
    """
    最佳旋转算法；p和q应该是尺寸 (N，3) 的三维点集合，其中N是点数。返回q2p的旋转平移变换
    :param p: 三维点集p
    :param q: 三维点集q
    :return: 返回一个旋转矩阵，平移向量，使得 p = (R_matrix @ q.T + Trans_vec.reshape((3, 1))).T，从q的坐标系转换到p的坐标系
    """

    # 计算两个点集的质心。
    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)

    # 通过减去它们的质心来使点集居中。
    p_centered = p - centroid_p
    q_centered = q - centroid_q

    # 计算中心点集的协方差矩阵。
    cov = p_centered.T.dot(q_centered)

    # 计算协方差矩阵的奇异值分解。
    U, S, V = np.linalg.svd(cov)

    # 通过取U和V矩阵的点积来计算旋转矩阵。
    r_matrix = U.dot(V)

    # 通过取质心的差异来计算平移矢量
    # 两个点集，并将其乘以旋转矩阵。
    trans_vec = centroid_p - r_matrix.dot(centroid_q)

    return r_matrix, trans_vec


def get_transform_matrix(r_matrix, trans_vec):
    """
    将旋转矩阵和平移向量组合成一个变换矩阵。
    :param r_matrix: 3x3 旋转矩阵
    :param trans_vec: 1x3 平移向量
    :return transform_matrix: 4x4 变换矩阵
    """
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = r_matrix
    transform_matrix[:3, 3] = trans_vec.reshape(1, 3)
    return transform_matrix


def apply_transform_matrix(transform_matrix, q):
    """
    :param transform_matrix: 4x4 变换矩阵
    :param q: Nx3 三维点集 q
    :return: 应用了transform_matrix后的点集 q
    """
    return np.dot(transform_matrix, np.vstack([q.T, np.ones(len(q))])).T[:, :3]


def apply_RT(r_matrix, trans_vec, q):
    """
    :param r_matrix: 3x3 旋转矩阵
    :param trans_vec: 1x3 平移向量
    :param q: Nx3 点集 q
    :return: (r_matrix @ q.T + trans_vec.reshape((3, 1))).T
    """
    return (r_matrix @ q.T + trans_vec.reshape((3, 1))).T


def get_r_mat_from_r_vec(rotation_vector):
    # 旋转向量
    # rotation_vector = np.array([0.1, 0.2, 0.3])
    # 创建 Rotation 对象
    rotation = Rotation.from_rotvec(rotation_vector)
    # 获取旋转矩阵
    return rotation.as_matrix()


def rmse(predictions, targets):
    """
    计算均方根误差 (RMSE)

    参数：
    - predictions: 预测值的集合
    - targets: 实际值的集合

    返回值：
    - rmse_value: RMSE 值
    """
    rmse_value = np.sqrt(((predictions - targets) ** 2).mean())
    return rmse_value


def reshape_to_vector(X):
    # Reshape X (4x4 matrix) to a 1D vector
    return X.flatten()


def reshape_to_matrix(X_vector):
    # Reshape a 1D vector to a 4x4 matrix
    return X_vector.reshape(4, 4)


def objective_function(X_vector, A, B):
    X_matrix = reshape_to_matrix(X_vector)
    N = A.shape[0]
    residuals = []

    for i in range(N):
        for j in range(N):
            if i != j:
                diff = np.dot(A[i], np.dot(X_matrix, B[i])) - np.dot(A[j], np.dot(X_matrix, B[j]))
                residuals.extend(diff.flatten())

    return np.array(residuals)


# def optimize_X(A, B):
#     N = A.shape[0]
#
#     # Initialize X as a 4x4 identity matrix and reshape it to a 1D vector
#     X_initial = reshape_to_vector(np.identity(4))
#
#     # Use least_squares function for optimization
#     result = least_squares(objective_function, X_initial, args=(A, B), method='trf')
#
#     # Reshape the optimized X from a 1D vector to a 4x4 matrix
#     X_optimal = reshape_to_matrix(result.x)
#
#     return X_optimal


def hand_eye_calibration_eye_to_hand(Poses_b2e, Poses_t2c, cali_method=0):
    """
    :param Poses_b2e: base to end 机械臂基座到末端的旋转平移矩阵 N*4*4 numpy array
        换言之如果获取的是机械臂末端的的坐标 h = [x, y, z, rx, ry, rz]
        得到的 Trans_e2b = transform.get_transform_matrix(transform.get_r_mat_from_r_vec(np.array(h[3:])),np.array(h[:3]).reshape(1, 3))
        需要采用 np.linalg.inv 获得 base to end 的旋转平移关系
        参考了 https://blog.csdn.net/qq_40016998/article/details/121099134
        官网文档 https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
        注意 cv2.calibrateHandEye 的输入描述是 eye-in-hand configuration，需要改成 eye-to-hand configuration 的参数列表
        cv2.calibrateHandEye 前两个的输入是 base to end 的 R t！！！！！！！！

        注意 calibrateHandEye()

    :param Poses_t2c: tool to camera 机械臂末端绑定的标定板（工具）到相机坐标系的旋转平移矩阵 N*4*4 numpy array
        换言之标定板在相机坐标系下的坐标 t = [x, y, z, rx, ry, rz]
        Poses_t2c.append(transform.get_transform_matrix(transform.get_r_mat_from_r_vec(np.array(t[3:])),np.array(t[:3]).reshape(1, 3)))
    :param cali_method
        cv2.CALIB_HAND_EYE_TSAI = 0
        cv2.CALIB_HAND_EYE_PARK = 1
        cv2.CALIB_HAND_EYE_HORAUD = 2
        cv2.CALIB_HAND_EYE_ANDREFF = 3
        cv2.CALIB_HAND_EYE_DANIILIDIS = 4
    :return:
        Trans_c2b: camera to base 相机到基座的旋转平移关系，对 Trans_c2b 取逆可以得到 Trans_b2c 即为相机坐标系下基座的坐标 4*4 numpy array
    """
    R_c2b, T_c2b = cv2.calibrateHandEye(Poses_b2e[:, :3, :3],
                                        Poses_b2e[:, :3, 3],
                                        Poses_t2c[:, :3, :3],
                                        Poses_t2c[:, :3, 3],
                                        method=cali_method)
    return get_transform_matrix(R_c2b, T_c2b)


# Example usage:
# A_matrices = np.array([...]) # Replace with actual data
# B_matrices = np.array([...]) # Replace with actual data
# X = tsai_calibration_revised_fixed_v2(A_matrices, B_matrices)
# print(X)


if __name__ == '__main__':
    import csv

    FILE_NAME = "../data/20240112-data/240112-bone-data.csv"
    # ENCODING_METHOD = "utf-8"  # UTF-8
    ENCODING_METHOD = "utf-8-sig"  # UTF-8-BOM
    measured_coordinates = []  # 你已经测量的定位系统下的三维坐标
    design_coordinates = []  # 体素坐标 单位 mm 找了像素点，作为x，y坐标，层数为z，分别与对应尺度因子相乘得到实际CT坐标系下坐标值
    # 打开CSV文件
    with open(FILE_NAME, 'r', newline='', encoding=ENCODING_METHOD) as file:
        # 创建CSV字典读取器

        csv_reader = csv.DictReader(file)
        # 逐行读取数据
        for row in csv_reader:
            # row是一个字典，包含CSV文件中的一行数据
            new_pin_w = list([float(row['pin_w_x']), float(row['pin_w_y']), float(row['pin_w_z'])])
            new_CT_mm = list([float(row['CT_mm_x']), float(row['CT_mm_y']), float(row['CT_mm_z'])])
            measured_coordinates.append(new_pin_w)
            design_coordinates.append(new_CT_mm)

    measured_coordinates = np.asarray(measured_coordinates)
    design_coordinates = np.asarray(design_coordinates)
    R_matrix, Trans_vec = kabsch(measured_coordinates, design_coordinates)
    print("R:\n", R_matrix, "\nT:\n", Trans_vec)
    print("MEASURE ")
    print(measured_coordinates)
    print("TRANFORM R@q +T")
    print(apply_RT(R_matrix, Trans_vec, design_coordinates))
    print("TRANFORM MATRIX")
    print(apply_transform_matrix(get_transform_matrix(R_matrix, Trans_vec), design_coordinates))

    # tsai_calibration()
