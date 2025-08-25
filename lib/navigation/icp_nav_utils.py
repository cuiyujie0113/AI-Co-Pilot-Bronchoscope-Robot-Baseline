"""ICP 导航辅助函数 (阶段1)

目标: 为后续在 onlineSimulation 中集成 "预测深度 vs 参考深度" 的 ICP 配准与位姿增量估计提供基础工具.

当前只做: 深度图 -> 点云, ICP 求解刚体变换, 以及简单的旋转分解(总角度). 先不直接映射到 pitch / yaw, 等确认坐标系后再细化。

使用方式(临时测试示例):
    import cv2, open3d as o3d, numpy as np
    from lib.navigation.icp_nav_utils import depth_uint8_to_meters, depth_to_pointcloud, icp_align
    d_pred = cv2.imread('pred_depth_000000.jpg', cv2.IMREAD_GRAYSCALE)
    d_gt   = cv2.imread('depth_000000.jpg', cv2.IMREAD_GRAYSCALE)
    dm_pred = depth_uint8_to_meters(d_pred)
    dm_gt   = depth_uint8_to_meters(d_gt)
    fx = fy = 100.0  # 占位，需与渲染/相机真实内参一致
    cx = cy = 100.0
    pc_pred = depth_to_pointcloud(dm_pred, fx, fy, cx, cy)
    pc_gt   = depth_to_pointcloud(dm_gt, fx, fy, cx, cy)
    T, rmse, corr = icp_align(pc_pred, pc_gt, voxel_size=0.002)
    print('T=\n', T, 'rmse=', rmse, 'corr=', corr)

TODO (后续阶段):
1. 根据 pybullet 中 pitch/yaw 角定义，将 T 的旋转部分映射为增量 pitch/yaw。
2. 融合 ahead ground-truth point (look-ahead) 的方向矢量，形成新的姿态目标。
3. 将平移增量融合到当前位置 t (或仅用于角度微调)。
4. 加入鲁棒/降噪策略 (跳过点数不足 / rmse 过大 / 协方差滤波)。
"""
from __future__ import annotations
import numpy as np
import open3d as o3d
from typing import Tuple

def depth_uint8_to_meters(depth_u8: np.ndarray, max_depth: float = 0.5) -> np.ndarray:
    """将 0~255 uint8 深度图转换为米 (与项目约定: 255 -> max_depth)."""
    if depth_u8 is None:
        raise ValueError('depth_uint8_to_meters: 输入为空')
    return depth_u8.astype(np.float32) / 255.0 * max_depth

def depth_to_pointcloud(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float, min_depth: float = 1e-4) -> o3d.geometry.PointCloud:
    """将深度图(米)投影为点云 (相机坐标: x 向右, y 向下, z 向前; 后续与仿真坐标差异需单独处理).

    说明: onlineSimulation 中相机坐标系注释存在 Z前 / Y下 的不同说法, 这里暂按常见 pinhole 模型。
    后续若需转换到仿真坐标, 可在此添加旋转矩阵变换。
    """
    h, w = depth_m.shape
    mask = depth_m > min_depth
    ys, xs = np.where(mask)
    zs = depth_m[ys, xs]
    xs_cam = (xs - cx) * zs / fx
    ys_cam = (ys - cy) * zs / fy
    pts = np.stack([xs_cam, ys_cam, zs], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def icp_align(src: o3d.geometry.PointCloud,
              tgt: o3d.geometry.PointCloud,
              voxel_size: float = 0.002) -> Tuple[np.ndarray, float, int]:
    """对齐 src->tgt, 返回 (4x4 变换矩阵, inlier RMSE, 对应点数)."""
    src_down = src.voxel_down_sample(voxel_size)
    tgt_down = tgt.voxel_down_sample(voxel_size)
    if len(src_down.points) < 50 or len(tgt_down.points) < 50:
        raise ValueError('点数不足无法ICP: src={} tgt={}'.format(len(src_down.points), len(tgt_down.points)))
    radius = voxel_size * 2
    src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    reg = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down, voxel_size * 1.5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return reg.transformation, reg.inlier_rmse, len(reg.correspondence_set)

def rotation_angle_from_T(T: np.ndarray) -> float:
    """返回整体旋转角度(度)."""
    R = T[:3, :3]
    trace = np.clip(np.trace(R), -1.0, 3.0)
    ang = np.degrees(np.arccos((trace - 1.0) / 2.0))
    return float(ang)

def decompose_pitch_yaw_approx(T: np.ndarray, order: str = 'XYZ') -> Tuple[float, float]:
    """提取近似 pitch / yaw (度).

    假设相机坐标: X 右, Y 下, Z 前(常见图像坐标), pitch = 绕 X 旋转(抬头正?), yaw = 绕 Y 或 Z 需区分。
    由于当前尚未最终确认仿真使用的欧拉顺序, 这里提供两种常见情况:
    order='XYZ': 先绕 X(pitch), 再 Y(yaw), 再 Z(roll)
    order='ZYX': 先绕 Z(yaw), 再 Y(pitch), 再 X(roll)

    注意: 仅用于小角度对齐和调试, 最终请以仿真定义为准。
    返回: (pitch_deg, yaw_deg)
    """
    R = T[:3, :3].astype(np.float64)
    eps = 1e-9
    if order.upper() == 'XYZ':
        # R = Rz * Ry * Rx (这里逆序, 但我们只近似反解 pitch(X), yaw(Y))
        # pitch (around X) from Ryxz? 为简化, 采用标准公式
        yaw = np.degrees(np.arctan2(-R[2, 0], R[2, 2] + eps))  # around Y
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2] + eps))  # around X (近似), 这部分存在歧义
    elif order.upper() == 'ZYX':
        # 常见航空: R = Rx * Ry * Rz (相机计算库中 SciPy Rotation 'ZYX' 表示 yaw(Z)-pitch(Y)-roll(X))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0] + eps))  # yaw around Z
        pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2) + eps))  # pitch around Y
    else:
        # 回退到原始简单法 (Z 作为 yaw, X 作为 pitch)
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0] + eps))
        pitch = np.degrees(np.arctan2(-R[2, 1], R[2, 2] + eps))
    return float(pitch), float(yaw)

__all__ = [
    'depth_uint8_to_meters',
    'depth_to_pointcloud',
    'icp_align',
    'rotation_angle_from_T',
    'decompose_pitch_yaw_approx',
    'R_S_to_C',
    'R_C_to_S',
    'convert_R_C_to_S',
    'extract_yaw_pitch_from_R_S',
    'fuse_icp_delta_into_pitch_yaw',
    'icp_step'
]

# ================== 坐标系转换 (S 与 C) ==================
# S: 控制/仿真使用 (X右 Y前 Z上) —— 代码中 p.getQuaternionFromEuler([pitch,0,yaw]) 的空间
# C: 相机/视觉使用 (X右 Y下 Z前) —— 深度投影/ICP 使用
# 关系: C = RotX(+90°) * S  ->  R_S_to_C = RotX(+90°)
R_S_to_C = np.array([[1,0,0],
                     [0,0,-1],
                     [0,1,0]], dtype=np.float32)
R_C_to_S = R_S_to_C.T  # RotX(-90°)

def convert_R_C_to_S(R_C: np.ndarray) -> np.ndarray:
    """将 C 系旋转矩阵转换到 S 系."""
    return R_C_to_S @ R_C @ R_S_to_C

def extract_yaw_pitch_from_R_S(R_S: np.ndarray) -> Tuple[float, float]:
    """从 S 系旋转矩阵稳健提取 (pitch_deg, yaw_deg)。

    约定 S 系: X右 Y前 Z上；yaw 绕 Z，pitch 绕 X；roll≈0。
    步骤: 先解 yaw_Z，再用 Rz(-yaw) 去耦后解 pitch_X。
    """
    eps = 1e-9
    # 1) yaw around Z: use standard formula yaw = atan2(r21, r11)
    yaw_rad = np.arctan2(R_S[1, 0], R_S[0, 0] + eps)
    # 2) remove yaw
    cy, sy = np.cos(-yaw_rad), np.sin(-yaw_rad)
    Rz_neg_yaw = np.array([[cy, -sy, 0],
                           [sy,  cy, 0],
                           [ 0,   0, 1]], dtype=R_S.dtype)
    R_no_yaw = Rz_neg_yaw @ R_S
    # 3) pitch around X: with roll≈0, R_no_yaw[2,1] = sin(pitch), R_no_yaw[2,2] = cos(pitch)
    pitch_rad = np.arctan2(R_no_yaw[2, 1], R_no_yaw[2, 2] + eps)
    return float(np.degrees(pitch_rad)), float(np.degrees(yaw_rad))

def fuse_icp_delta_into_pitch_yaw(pitch_deg: float,
                                  yaw_deg: float,
                                  R_delta_C: np.ndarray,
                                  alpha: float = 1.0,
                                  rmse: float | None = None,
                                  rmse_thresh: float = 0.01,
                                  max_angle_deg: float = 8.0) -> Tuple[float, float]:
    """融合 ICP 旋转增量到当前 (pitch,yaw) (单位: 度)。

    步骤:
      1. 将 R_delta_C (C 系) 转换到 S 系 -> R_delta_S。
      2. 从 R_delta_S 近似提取 (dpitch, dyaw)。
      3. 角度裁剪 (max_angle_deg)。
      4. 质量门限: rmse 若提供且 > rmse_thresh 则忽略。
      5. 与当前姿态线性融合: new = old + alpha * d.

    注意: 线性融合适用于小角度 (几度以内)。后续可替换为四元数 slerp。
    """
    if rmse is not None and rmse > rmse_thresh:
        return pitch_deg, yaw_deg  # ICP 质量差, 放弃更新
    # 1. 转换到 S 系
    R_delta_S = convert_R_C_to_S(R_delta_C)
    # 2. 提取增量 (假设小角度)
    dpitch, dyaw = extract_yaw_pitch_from_R_S(R_delta_S)
    # 3. 限幅
    dpitch = float(np.clip(dpitch, -max_angle_deg, max_angle_deg))
    dyaw = float(np.clip(dyaw, -max_angle_deg, max_angle_deg))
    # 4. 融合
    new_pitch = pitch_deg + alpha * dpitch
    new_yaw = yaw_deg + alpha * dyaw
    return new_pitch, new_yaw

def icp_step(depth_pred_u8: np.ndarray,
                         depth_gt_u8: np.ndarray,
                         fx: float, fy: float, cx: float, cy: float,
                         pitch_deg: float, yaw_deg: float,
                         alpha: float = 0.5,
                         rmse_thresh: float = 0.01,
                         voxel_size: float = 0.002,
                         max_angle_deg: float = 8.0,
                         max_depth_m: float = 0.5) -> Tuple[float, float, dict]:
        """单步 ICP 融合：由预测/GT 深度图(uchar)计算 ICP 增量并更新 (pitch,yaw)。

        返回: (new_pitch_deg, new_yaw_deg, info)
            info: {
                'T': 4x4, 'rmse': float, 'corr': int,
                'dpitch': float, 'dyaw': float,
                'R_delta_S': 3x3, 'R_delta_C': 3x3
            }
        """
        # 1) 转米
        d_pred_m = depth_uint8_to_meters(depth_pred_u8, max_depth=max_depth_m)
        d_gt_m = depth_uint8_to_meters(depth_gt_u8, max_depth=max_depth_m)
        # 2) 点云 (C 系)
        pc_pred = depth_to_pointcloud(d_pred_m, fx, fy, cx, cy)
        pc_gt = depth_to_pointcloud(d_gt_m, fx, fy, cx, cy)
        # 3) ICP (pred -> gt)
        T, rmse, corr = icp_align(pc_pred, pc_gt, voxel_size=voxel_size)
        R_delta_C = T[:3, :3]
        # 4) 转 S 系并提取
        R_delta_S = convert_R_C_to_S(R_delta_C)
        dpitch, dyaw = extract_yaw_pitch_from_R_S(R_delta_S)
        # 5) 融合 (带阈值与限幅)
        if rmse > rmse_thresh:
                new_pitch, new_yaw = pitch_deg, yaw_deg
        else:
                dpitch = float(np.clip(dpitch, -max_angle_deg, max_angle_deg))
                dyaw = float(np.clip(dyaw, -max_angle_deg, max_angle_deg))
                new_pitch = pitch_deg + alpha * dpitch
                new_yaw = yaw_deg + alpha * dyaw
        info = {
                'T': T,
                'rmse': rmse,
                'corr': corr,
                'dpitch': dpitch,
                'dyaw': dyaw,
                'R_delta_S': R_delta_S,
                'R_delta_C': R_delta_C,
        }
        return float(new_pitch), float(new_yaw), info
