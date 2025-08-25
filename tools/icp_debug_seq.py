import argparse
import os
import cv2
import numpy as np
from lib.navigation.icp_nav_utils import icp_step

FX0 = FY0 = 175.0 / 1.008
CX0 = CY0 = 200.0


def compute_intrinsics_for_size(w: int, h: int):
    sx = w / 400.0
    sy = h / 400.0
    fx = FX0 * sx
    fy = FY0 * sy
    cx = CX0 * sx
    cy = CY0 * sy
    return fx, fy, cx, cy


def imread_u8(path):
    import numpy as _np
    import cv2 as _cv
    img = _cv.imdecode(_np.fromfile(path, dtype=_np.uint8), _cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def main():
    parser = argparse.ArgumentParser(description="Run ICP across a sequence, accumulating pitch/yaw.")
    parser.add_argument("--centerline", type=str, default="centerline_00_siliconmodel3_Centerline_model")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to iterate from index 0")
    parser.add_argument("--pred-root", type=str, default=os.path.join("data", "unet_predictions_jitter"))
    parser.add_argument("--gt-root", type=str, default=os.path.join("data", "all_centerline_datasets"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rmse-thresh", type=float, default=0.012)
    parser.add_argument("--voxel", type=float, default=0.002)
    parser.add_argument("--max-angle", type=float, default=6.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()

    pitch, yaw = args.pitch, args.yaw
    for i in range(args.frames):
        idx = f"{i:06d}"
        pred_path = os.path.join(args.pred_root, args.centerline, "depth_images", f"depth_{idx}.jpg")
        gt_path = os.path.join(args.gt_root, args.centerline, "depth_images", f"depth_{idx}.jpg")
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"[stop] missing at {i}: pred?{os.path.exists(pred_path)} gt?{os.path.exists(gt_path)}")
            break
        d_pred = imread_u8(pred_path)
        d_gt = imread_u8(gt_path)
        if d_pred.shape != d_gt.shape:
            d_gt = cv2.resize(d_gt, (d_pred.shape[1], d_pred.shape[0]), interpolation=cv2.INTER_AREA)
        h, w = d_pred.shape[:2]
        fx, fy, cx, cy = compute_intrinsics_for_size(w, h)
        pitch, yaw, info = icp_step(d_pred, d_gt, fx, fy, cx, cy,
                                    pitch_deg=pitch, yaw_deg=yaw,
                                    alpha=args.alpha, rmse_thresh=args.rmse_thresh,
                                    voxel_size=args.voxel, max_angle_deg=args.max_angle)
        print(f"[{i}] rmse={info['rmse']:.5f} corr={info['corr']} dp={info['dpitch']:.3f} dy={info['dyaw']:.3f} -> p={pitch:.3f} y={yaw:.3f}")


if __name__ == "__main__":
    main()
