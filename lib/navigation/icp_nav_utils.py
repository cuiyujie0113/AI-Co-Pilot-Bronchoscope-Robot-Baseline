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

def decompose_pitch_yaw_approx(T: np.ndarray) -> Tuple[float, float]:
    """简易近似: 假设旋转主要由绕 X(=pitch), Z(=yaw) 组成, 提取两个角度(度).

    注意: 需与 onlineSimulation 中的坐标系核对后才能正式使用。
    当前采用标准相机坐标假设: X 右, Y 下, Z 前; yaw 绕 Yaw(Z), pitch 绕 X.
    """
    R = T[:3, :3]
    eps = 1e-8
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0] + eps))
    pitch = np.degrees(np.arctan2(-R[2, 1], R[2, 2] + eps))
    return pitch, yaw

a__all__ = [
    'depth_uint8_to_meters',
    'depth_to_pointcloud',
    'icp_align',
    'rotation_angle_from_T',
    'decompose_pitch_yaw_approx'
]
