import argparse
import os
import cv2
import numpy as np
from lib.navigation.icp_nav_utils import icp_step, rotation_angle_from_T

# Base intrinsics for 400x400 rendering (project convention)
FX0 = FY0 = 175.0 / 1.008
CX0 = CY0 = 200.0


def compute_intrinsics_for_size(w: int, h: int):
    # Assumes original 400x400; scales intrinsics linearly to target size
    sx = w / 400.0
    sy = h / 400.0
    fx = FX0 * sx
    fy = FY0 * sy
    cx = CX0 * sx
    cy = CY0 * sy
    return fx, fy, cx, cy


def load_depth_u8(path: str):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def main():
    parser = argparse.ArgumentParser(description="Run ICP between predicted and GT depth images and print corrections.")
    parser.add_argument("--centerline", type=str, default="centerline_00_siliconmodel3_Centerline_model",
                        help="Centerline folder name under both roots")
    parser.add_argument("--idx", type=int, default=0, help="Frame index, e.g., 0 for depth_000000.jpg")
    parser.add_argument("--pred-root", type=str,
                        default=os.path.join("data", "unet_predictions_jitter"),
                        help="Root directory for predicted depth folders")
    parser.add_argument("--gt-root", type=str,
                        default=os.path.join("data", "all_centerline_datasets"),
                        help="Root directory for GT depth folders")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend factor for angle fusion")
    parser.add_argument("--rmse-thresh", type=float, default=0.012, help="ICP rmse threshold to accept updates")
    parser.add_argument("--voxel", type=float, default=0.002, help="Voxel size for downsampling in meters")
    parser.add_argument("--max-angle", type=float, default=8.0, help="Clamp for per-step angle update (deg)")
    parser.add_argument("--pitch", type=float, default=0.0, help="Initial pitch (deg) in S-frame")
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw (deg) in S-frame")
    args = parser.parse_args()

    idx_str = f"{args.idx:06d}"
    pred_path = os.path.join(args.pred_root, args.centerline, "depth_images", f"depth_{idx_str}.jpg")
    gt_path = os.path.join(args.gt_root, args.centerline, "depth_images", f"depth_{idx_str}.jpg")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Pred depth not found: {pred_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT depth not found: {gt_path}")

    d_pred = load_depth_u8(pred_path)
    d_gt = load_depth_u8(gt_path)

    # If sizes differ, resize GT to pred's size using area interpolation (reasonable for depth)
    if d_pred.shape != d_gt.shape:
        d_gt = cv2.resize(d_gt, (d_pred.shape[1], d_pred.shape[0]), interpolation=cv2.INTER_AREA)
        print(f"[info] Resized GT from to {d_gt.shape} to match pred {d_pred.shape}")

    h, w = d_pred.shape[:2]
    fx, fy, cx, cy = compute_intrinsics_for_size(w, h)

    new_pitch, new_yaw, info = icp_step(
        depth_pred_u8=d_pred,
        depth_gt_u8=d_gt,
        fx=fx, fy=fy, cx=cx, cy=cy,
        pitch_deg=args.pitch, yaw_deg=args.yaw,
        alpha=args.alpha,
        rmse_thresh=args.rmse_thresh,
        voxel_size=args.voxel,
        max_angle_deg=args.max_angle,
        max_depth_m=0.5,
    )

    T = info.get('T')
    ang = rotation_angle_from_T(T)
    print("=== ICP result ===")
    print(f"centerline: {args.centerline}  idx: {args.idx}")
    print(f"image size: {w}x{h}   fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
    print(f"rmse: {info['rmse']:.6f}  corr: {info['corr']}  rot|deg: {ang:.3f}")
    print(f"dpitch: {info['dpitch']:.4f}  dyaw: {info['dyaw']:.4f}  (clamped then blended)")
    print(f"old(p,y)=({args.pitch:.3f},{args.yaw:.3f})  ->  new(p,y)=({new_pitch:.3f},{new_yaw:.3f})")


if __name__ == "__main__":
    main()
