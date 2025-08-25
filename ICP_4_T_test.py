import cv2, open3d as o3d
import numpy as np
from lib.navigation.icp_nav_utils import depth_uint8_to_meters, depth_to_pointcloud, icp_align, rotation_angle_from_T, decompose_pitch_yaw_approx

gt = cv2.imread(r'data\all_centerline_datasets\centerline_00_siliconmodel3_Centerline_model\depth_images\depth_000020.jpg', cv2.IMREAD_GRAYSCALE)
pred = cv2.imread(r'data\unet_predictions_jitter\centerline_00_siliconmodel3_Centerline_model\depth_images\depth_000020.jpg', cv2.IMREAD_GRAYSCALE)

dm_gt = depth_uint8_to_meters(gt)
dm_pred = depth_uint8_to_meters(pred)

fx = fy = 175/1.008  # 参考 onlineSimulation 里 IntrinsicsCamera 设置 (后续确认 cx, cy)
cx = cy = 200        # onlineSimulation 用 400x400 渲染后再 resize 200, 这里暂用 200 做测试 (需统一)
pc_gt = depth_to_pointcloud(dm_gt, fx, fy, cx, cy)
pc_pred = depth_to_pointcloud(dm_pred, fx, fy, cx, cy)

T, rmse, corr = icp_align(pc_pred, pc_gt)
print('T=', T, 'rmse=', rmse, 'corr=', corr)
print('rot_angle(deg)=', rotation_angle_from_T(T))
print('approx pitch,yaw=', decompose_pitch_yaw_approx(T))

pitch_xyz, yaw_xyz = decompose_pitch_yaw_approx(T, 'XYZ')
pitch_zyx, yaw_zyx = decompose_pitch_yaw_approx(T, 'ZYX')
print('XYZ pitch,yaw=', pitch_xyz, yaw_xyz)
print('ZYX pitch,yaw=', pitch_zyx, yaw_zyx)