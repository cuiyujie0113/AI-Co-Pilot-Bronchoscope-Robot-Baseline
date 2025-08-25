import argparse
import os
import glob
from typing import Tuple, List

import numpy as np
import cv2
import torch
import pybullet as p
import trimesh
from pyrender import IntrinsicsCamera, MetallicRoughnessMaterial, Mesh, Node, Scene, SpotLight, OffscreenRenderer

from lib.navigation.icp_nav_utils import icp_step
from unet_depth_pipeline import load_model as unet_load_model, create_viz


# 基准内参（400x400 渲染约定）
FX0 = FY0 = 175.0 / 1.008
CX0 = CY0 = 200.0


def compute_intrinsics_for_size(w: int, h: int) -> Tuple[float, float, float, float]:
    sx = w / 400.0
    sy = h / 400.0
    fx = FX0 * sx
    fy = FY0 * sy
    cx = CX0 * sx
    cy = CY0 * sy
    return fx, fy, cx, cy


def imread_u8_gray(path: str) -> np.ndarray:
    arr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return arr

def imread_rgb(path: str) -> np.ndarray:
    """Unicode 安全读取 RGB 图 (np.uint8, HxWx3, RGB)."""
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def parse_dataset_info(txt_path: str):
    """
    读取 camera_info/dataset_info.txt
    返回：
    - pos_list: (N,3) np.array [x,y,z]
    - pitch_list: (N,) degrees in S-frame
    - yaw_list: (N,) degrees in S-frame
    - dist_list: (N,) 累积弧长（单位与数据一致，通常米）
    """
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    pitchs: List[float] = []
    yaws: List[float] = []
    dists: List[float] = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 7:
                continue
            # idx = int(parts[0])  # 未使用
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            pitch = float(parts[4])
            yaw = float(parts[5])
            dist = float(parts[6])
            xs.append(x)
            ys.append(y)
            zs.append(z)
            pitchs.append(pitch)
            yaws.append(yaw)
            dists.append(dist)
    pos = np.stack([xs, ys, zs], axis=1)
    return pos, np.array(pitchs), np.array(yaws), np.array(dists)


def yaw_pitch_from_vector_S(v: np.ndarray) -> Tuple[float, float]:
    """根据 S 系(右手: X右/Y前/Z上)的方向向量 v，计算 yaw(Z轴) 与 pitch(X轴)，单位: 度。
    与仓库 onlineSimulation 中的实现保持一致：先 yaw 后 pitch。
    """
    vx, vy, vz = v.astype(float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return 0.0, 0.0
    pitch = np.arcsin(vz / n)
    # yaw 分支（参考仓库写法）
    denom = np.sqrt(vx * vx + vy * vy) + 1e-12
    if vx > 0:
        yaw = -np.arccos(vy / denom)
    else:
        yaw = np.arccos(vy / denom)
    return float(np.degrees(yaw)), float(np.degrees(pitch))


def forward_from_yaw_pitch_S(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """从 S 系 yaw/pitch(度) 计算相机前向单位向量。
    约定：先绕 Z 轴 yaw，再绕 X 轴 pitch，基向量为 +Y 前。
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    # Rz(yaw) * Rx(pitch) * [0,1,0]
    # 先绕 Z
    c = np.cos(yaw)
    s = np.sin(yaw)
    v = np.array([-s, c, 0.0])  # Rz(yaw) * [0,1,0]
    # 再绕 X
    cx = np.cos(pitch)
    sx = np.sin(pitch)
    vx, vy, vz = v[0], v[1] * cx - v[2] * sx, v[1] * sx + v[2] * cx
    out = np.array([vx, vy, vz], dtype=float)
    n = np.linalg.norm(out)
    if n > 1e-12:
        out /= n
    return out


class PyrenderContext:
    """轻量渲染器：复用项目的 pyrender 设置，实时渲染 RGB/Depth。
    - 模型: airways/AirwayHollow_{tag}_simUV.obj, scale=0.01, rotation=RotX(+90°), translation=[0,0,5]
    - 相机: 内参基于 400x400，输出可缩放到 200x200 做 ICP
    - 姿态: 从 S 系 yaw/pitch(度) + 位置 t(米) 转换为渲染 pose。
    """
    def __init__(self, model_tag: str, render_size: int = 400, output_size: int = 200, light_intensity: float = 0.3,
                 auto_exposure: bool = True, target_mean: float = 140.0, tol: float = 20.0, max_iter: int = 50):
        self.render_size = render_size
        self.output_size = output_size
        self.light_intensity = light_intensity
        self.auto_exposure = auto_exposure
        self.target_mean = target_mean
        self.tol = tol
        self.max_iter = max_iter
        # 内参 (与项目约定一致)
        self.fx0 = 175.0 / 1.008
        self.fy0 = 175.0 / 1.008
        self.cx0 = 200.0
        self.cy0 = 200.0
        s = output_size / render_size
        self.fx_out = self.fx0 * s
        self.fy_out = self.fy0 * s
        self.cx_out = self.cx0 * s
        self.cy_out = self.cy0 * s
        # 加载气道网格
        obj_path = os.path.join('airways', f'AirwayHollow_{model_tag}_simUV.obj')
        if not os.path.exists(obj_path):
            raise FileNotFoundError(obj_path)
        # 加载并确保为单一 Trimesh（部分 OBJ 会被解析为 Scene，需合并几何）
        try:
            geom = trimesh.load(obj_path)
        except Exception:
            geom = trimesh.load_mesh(obj_path)
        try:
            import trimesh as _tm
            if isinstance(geom, _tm.Scene):
                geom = _tm.util.concatenate(tuple(geom.geometry.values()))
        except Exception:
            pass
        self.tri_geom = geom
        material = MetallicRoughnessMaterial(
            metallicFactor=0.1,
            alphaMode='OPAQUE',
            roughnessFactor=0.7,
            baseColorFactor=[206/255,108/255,131/255,1],
        )
        self.mesh_node = Mesh.from_trimesh(self.tri_geom, material=material)
        # 变换与相机
        self.quaternion_model = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        self.t_model = np.array([0, 0, 5], dtype=np.float32)
        self.R_model = np.array(p.getMatrixFromQuaternion(self.quaternion_model), dtype=np.float32).reshape(3, 3)
        self.scale_model = 0.01  # 场景中将网格按 0.01 缩放（cm -> m）
        self.scene = Scene(bg_color=(0., 0., 0.))
        self.fuze_node = Node(mesh=self.mesh_node, scale=[0.01,0.01,0.01], rotation=self.quaternion_model, translation=self.t_model)
        self.scene.add_node(self.fuze_node)
        self.cam = IntrinsicsCamera(fx=self.fx0, fy=self.fy0, cx=self.cx0, cy=self.cy0, znear=1e-5)
        self.cam_node = self.scene.add(self.cam)
        self.renderer = OffscreenRenderer(viewport_width=render_size, viewport_height=render_size)
        # 距离查询缓存
        self._prox_q = None

    def _world_to_mesh_coords(self, x_world: np.ndarray) -> np.ndarray:
        # 先去除平移，再转到模型坐标，再换算到网格原尺度（除以 0.01）
        return (self.R_model.T @ (x_world - self.t_model)) / self.scale_model

    def check_inside_and_clearance(self, x_world: np.ndarray) -> tuple[bool, float]:
        """返回 (inside, clearance_m)。
        - 为稳健起见，inside 统一返回 True（多为非封闭管道，contains 结果不可靠）。
        - 优先使用最近表面距离，若计算失败，则返回一个很大的壁距，避免导航停滞。
        """
        x_mesh = self._world_to_mesh_coords(np.asarray(x_world, dtype=float))
        # 最近表面距离（网格单位）
        dist_mesh = None
        try:
            import trimesh.proximity as prox
            if self._prox_q is None:
                self._prox_q = prox.ProximityQuery(self.tri_geom)
            dists = self._prox_q.distance(x_mesh.reshape(1, 3))
            dist_mesh = float(dists[0])
        except Exception:
            # 若失败，给出一个很大的安全距离，避免卡住
            dist_mesh = 1e6
        # 转回米
        clearance_m = dist_mesh * self.scale_model
        inside = True
        return inside, clearance_m

    @staticmethod
    def _pose_from_S(yaw_deg: float, pitch_deg: float, t: np.ndarray) -> np.ndarray:
        # 映射到渲染用欧拉角（与 onlineSimulation 一致）
        pitch_r = np.radians(pitch_deg) + np.pi/2
        yaw_r = np.radians(yaw_deg)
        quat = p.getQuaternionFromEuler([pitch_r, 0, yaw_r])
        R = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float32).reshape(3,3)
        pose = np.eye(4, dtype=np.float32)
        pose[:3,:3] = R
        pose[:3, 3] = t.astype(np.float32)
        return pose

    def render_rgbd_S(self, yaw_deg: float, pitch_deg: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 重置场景节点并设置新姿态
        self.scene.clear()
        self.scene.add_node(self.fuze_node)
        pose = self._pose_from_S(yaw_deg, pitch_deg, t)
        light_intensity = self.light_intensity
        light = SpotLight(color=np.ones(3), intensity=light_intensity, innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
        ln = self.scene.add(light)
        cn = self.scene.add(self.cam)
        self.scene.set_pose(ln, pose)
        self.scene.set_pose(cn, pose)
        rgb, depth = self.renderer.render(self.scene)
        rgb = rgb[:, :, :3]
        # 自动曝光：二分搜索亮度到目标区间
        if self.auto_exposure:
            mean_intensity = float(np.mean(rgb))
            min_light, max_light = 0.001, 20.0
            it = 0
            while abs(mean_intensity - self.target_mean) > self.tol and it < self.max_iter:
                if mean_intensity > self.target_mean:
                    max_light = light_intensity
                else:
                    min_light = light_intensity
                light_intensity = (min_light + max_light) / 2.0
                # 重新渲染
                self.scene.clear(); self.scene.add_node(self.fuze_node)
                light = SpotLight(color=np.ones(3), intensity=light_intensity, innerConeAngle=0, outerConeAngle=np.pi/2, range=1)
                ln = self.scene.add(light); cn = self.scene.add(self.cam)
                self.scene.set_pose(ln, pose); self.scene.set_pose(cn, pose)
                rgb, depth = self.renderer.render(self.scene)
                rgb = rgb[:, :, :3]
                mean_intensity = float(np.mean(rgb))
                it += 1
        # 缩放到输出尺寸
        rgb_out = cv2.resize(rgb, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        # 深度与数据集一致的处理：将0填为0.5并裁剪
        depth[depth == 0] = 0.5
        depth = np.clip(depth, 0.0, 0.5)
        depth_u8 = (depth/0.5*255.0).astype(np.uint8)
        depth_out = cv2.resize(depth_u8, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        return rgb_out, depth_out


def main():
    parser = argparse.ArgumentParser(description="Centerline-guided navigation loop with ICP correction (no engine edits)")
    parser.add_argument("--centerline", type=str, default="centerline_00_siliconmodel3_Centerline_model")
    parser.add_argument("--gt-root", type=str, default=os.path.join("data", "all_centerline_datasets"))
    # 在线 UNet 深度估计
    parser.add_argument("--model-path", type=str, default=os.path.join("checkpoints","unet_depth_checkpoints","best_model.pth"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=200, help="UNet 推理分辨率")
    parser.add_argument("--frames", type=int, default=0, help="导航步数；<=0 表示自动运行直至遍历到中心线末端")
    parser.add_argument("--lookahead", type=int, default=5, help="Lookahead points ahead (indices)")
    parser.add_argument("--step-m", type=float, default=0.001, help="Move step in meters per iteration")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rmse-thresh", type=float, default=0.012)
    parser.add_argument("--voxel", type=float, default=0.002)
    parser.add_argument("--max-angle", type=float, default=6.0)
    parser.add_argument("--corr-min", type=int, default=0, help="If >0, require at least this many correspondences to accept ICP update")
    parser.add_argument("--reject-saturated", action="store_true", help="If set, reject ICP update when dp/dy hit max-angle clamp")
    parser.add_argument("--min-clearance", type=float, default=0.0015, help="最小壁距(米)，小于该值则收缩步长避免穿壁")
    # 自适应步长（可选）：根据 rmse/corr 调整前进步长
    parser.add_argument("--auto-step", action="store_true", help="启用基于配准质量的自适应步长")
    parser.add_argument("--step-scale-max", type=float, default=3.0, help="自适应步长的最大放大倍数")
    parser.add_argument("--rmse-good", type=float, default=0.003, help="判定配准很好的 rmse 阈值")
    parser.add_argument("--corr-good", type=int, default=2000, help="判定配准很好的对应点数量阈值")
    parser.add_argument("--start-dx", type=float, default=0.0, help="Initial real pose offset X (m)")
    parser.add_argument("--start-dy", type=float, default=0.0, help="Initial real pose offset Y (m)")
    parser.add_argument("--start-dz", type=float, default=0.0, help="Initial real pose offset Z (m)")
    parser.add_argument("--start-dyaw", type=float, default=0.0, help="Initial real yaw offset (deg)")
    parser.add_argument("--start-dpitch", type=float, default=0.0, help="Initial real pitch offset (deg)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-csv", type=str, default="", help="Optional CSV path to log per-step metrics")
    parser.add_argument("--viz-dir", type=str, default="", help="可选：保存每步 RGB/GT/Pred 可视化的目录")
    parser.add_argument("--make-video", action="store_true", help="结束后将可视化序列导出为视频")
    parser.add_argument("--fps", type=int, default=24, help="导出视频的帧率")
    parser.add_argument("--vid-stride", type=int, default=1, help="导出视频时的帧间隔(>=1，越大越快)")
    parser.add_argument("--video-out", type=str, default="", help="视频输出路径（留空则保存到 viz 目录 nav_run.mp4）")
    args = parser.parse_args()

    # 路径与信息文件（用于读取中心线位姿，不再读取 GT 深度图像）
    cen_dir = os.path.join(args.gt_root, args.centerline)
    info_txt = os.path.join(cen_dir, "camera_info", "dataset_info.txt")
    if not os.path.exists(info_txt):
        raise FileNotFoundError(info_txt)

    pos_gt, pitch_gt, yaw_gt, dist_gt = parse_dataset_info(info_txt)
    N = pos_gt.shape[0]
    if N == 0:
        raise RuntimeError("Empty centerline info")

    # 真实相机初始位姿 = 参考起点 + 偏差
    start_idx = 0
    pos_real = pos_gt[start_idx].copy()
    pos_real += np.array([args.start_dx, args.start_dy, args.start_dz], dtype=float)
    yaw_real = float(yaw_gt[start_idx] + args.start_dyaw)
    pitch_real = float(pitch_gt[start_idx] + args.start_dpitch)

    # 简单最近点索引（线性搜索，N<=数百足够；如需更快可替换KDTree）
    def nearest_index(p: np.ndarray) -> int:
        d2 = np.sum((pos_gt - p.reshape(1, 3)) ** 2, axis=1)
        return int(np.argmin(d2))

    # 解析模型 tag (如 centerline_00_siliconmodel3_Centerline_model -> siliconmodel3)
    try:
        parts = args.centerline.split('_')
        # 典型: centerline_00_siliconmodel3_Centerline_model -> 第3段是 siliconmodel3
        tag = parts[2] if len(parts) >= 3 else 'siliconmodel3'
    except Exception:
        tag = 'siliconmodel3'

    # 加载 UNet 深度模型
    device = torch.device(args.device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"未找到 UNet 模型: {args.model_path}")
    unet_model = unet_load_model(args.model_path, device)

    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    # 初始化渲染器
    renderer = PyrenderContext(model_tag=tag, render_size=400, output_size=200, light_intensity=0.3)

    step = 0
    while True:
        i = nearest_index(pos_real)
        # 正向（起点 -> 终点）参考索引
        j = min(i + args.lookahead, N - 1)

        # lookahead 方向与目标角
        v_des = pos_gt[j] - pos_gt[i]
        yaw_tar, pitch_tar = yaw_pitch_from_vector_S(v_des)

        # 实时渲染：当前相机视角 RGB（用于 UNet 预测深度）
        rgb_frame, _ = renderer.render_rgbd_S(yaw_real, pitch_real, pos_real)
        rgb_resized = cv2.resize(rgb_frame, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        rgb_t = torch.from_numpy((rgb_resized.astype(np.float32)/255.0)).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_t = unet_model(rgb_t)
        pred_depth = pred_t.squeeze().detach().cpu().numpy()  # [0,0.5]
        d_pred = (np.clip(pred_depth, 0.0, 0.5)/0.5*255.0).astype(np.uint8)

        # 实时渲染：参考视角深度（按 lookahead 目标方向定位参考相机）
        ref_pos = pos_gt[j]
        _, d_gt = renderer.render_rgbd_S(yaw_tar, pitch_tar, ref_pos)

        h, w = d_pred.shape[:2]
        fx, fy, cx, cy = compute_intrinsics_for_size(w, h)

        # 先以 lookahead 角作为当前角，进行 ICP 微校正（对比“当前帧Pred” vs “参考视角GT”）
        pitch_cmd, yaw_cmd, info = icp_step(
            depth_pred_u8=d_pred,
            depth_gt_u8=d_gt,
            fx=fx, fy=fy, cx=cx, cy=cy,
            pitch_deg=pitch_tar, yaw_deg=yaw_tar,
            alpha=args.alpha,
            rmse_thresh=args.rmse_thresh,
            voxel_size=args.voxel,
            max_angle_deg=args.max_angle,
            max_depth_m=0.5,
        )

        # 可选：根据 corr 最小值进行外部门控（icp_step 内部仅做 rmse 门控）
        rejected = False
        if args.corr_min > 0 and info.get('corr', 999999) < args.corr_min:
            rejected = True
            pitch_cmd, yaw_cmd = pitch_tar, yaw_tar

        # 可选：当 dp/dy 被 max-angle 限幅时，拒绝该次更新，回退到 lookahead 角
        if not rejected and args.reject_saturated:
            dp, dy = abs(info.get('dpitch', 0.0)), abs(info.get('dyaw', 0.0))
            if dp >= args.max_angle - 1e-6 or dy >= args.max_angle - 1e-6:
                rejected = True
                pitch_cmd, yaw_cmd = pitch_tar, yaw_tar

        # 推进一步（可选自适应步长）
        fwd = forward_from_yaw_pitch_S(yaw_cmd, pitch_cmd)
        step_len = float(args.step_m)
        if args.auto_step and not rejected:
            # rmse 越小越好 -> q1 在 [0,1]
            rmse = max(0.0, float(info.get('rmse', 0.0)))
            q1 = 1.0 - max(0.0, min(1.0, rmse / max(args.rmse_good, 1e-6)))
            # corr 越大越好 -> q2 在 [0,1]
            corr = float(info.get('corr', 0.0))
            q2 = max(0.0, min(1.0, corr / max(1.0, float(args.corr_good))))
            q = 0.5 * q1 + 0.5 * q2
            scale = 1.0 + q * max(0.0, args.step_scale_max - 1.0)
            step_len *= scale
        # 壁面安全：若新位置壁距不足，二分缩步
        target_clear = float(max(0.0, args.min_clearance))
        if target_clear > 0:
            lo, hi = 0.0, 1.0
            best = 0.0
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                cand = pos_real + fwd * (step_len * mid)
                inside, clr = renderer.check_inside_and_clearance(cand)
                ok = (clr >= target_clear) and inside
                # 放宽 inside 要求，至少保证壁距（避免 contains 不可用时卡死）
                if clr >= target_clear and (inside or clr >= target_clear):
                    best = mid
                    lo = mid
                else:
                    hi = mid
            if best > 1e-3:
                pos_real = pos_real + fwd * (step_len * best)
            # 若完全不安全，则不前进
        else:
            pos_real = pos_real + fwd * step_len
        yaw_real, pitch_real = yaw_cmd, pitch_cmd

        # 可视化（可选）：仅显示三个简单标题，且将中间与右侧顺序对调为 Pred 在中、GT 在右
        if args.viz_dir:
            import matplotlib.pyplot as plt
            gt_f = (d_gt.astype(np.float32)/255.0)*0.5
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            # 左：RGB
            ax[0].imshow(rgb_resized)
            ax[0].set_title('Current RGB', fontsize=10)
            ax[0].axis('off')
            # 中：Predicted Depth
            im1 = ax[1].imshow(pred_depth, cmap='plasma', vmin=0, vmax=0.5)
            ax[1].set_title('Predicted Depth', fontsize=10)
            ax[1].axis('off')
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
            # 右：Reference Depth
            im2 = ax[2].imshow(gt_f, cmap='plasma', vmin=0, vmax=0.5)
            ax[2].set_title('Reference Depth', fontsize=10)
            ax[2].axis('off')
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            fig.savefig(os.path.join(args.viz_dir, f"step_{step:04d}.png"), dpi=140, bbox_inches='tight')
            plt.close(fig)

        # 记录/打印
        if args.verbose:
            rej_str = " REJ" if rejected else ""
            print(
                f"[{step}] i={i} j={j} rmse={info['rmse']:.5f} corr={info.get('corr',-1)} "
                f"dp={info.get('dpitch',0):.3f} dy={info.get('dyaw',0):.3f}{rej_str} "
                f"-> pitch={pitch_cmd:.3f} yaw={yaw_cmd:.3f} pos=({pos_real[0]:.3f},{pos_real[1]:.3f},{pos_real[2]:.3f})"
            )

        # CSV 日志
        if args.log_csv:
            write_header = not os.path.exists(args.log_csv)
            with open(args.log_csv, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write('step,i,j,rmse,corr,dpitch,dyaw,pitch_cmd,yaw_cmd,posx,posy,posz,rejected\n')
                f.write(
                    f"{step},{i},{j},{info['rmse']:.6f},{info.get('corr',-1)},{info.get('dpitch',0):.6f},{info.get('dyaw',0):.6f},"
                    f"{pitch_cmd:.6f},{yaw_cmd:.6f},{pos_real[0]:.6f},{pos_real[1]:.6f},{pos_real[2]:.6f},{int(rejected)}\n"
                )

        # 终止：靠近末端
        if i >= N - 2:
            print(f"[done] reached near the end at idx {i}")
            break
        # 若设置了最大步数，则受其限制
        step += 1
        if args.frames > 0 and step >= args.frames:
            break

    # 导出视频（可选）
    if args.viz_dir and args.make_video:
        try:
            frames = sorted(glob.glob(os.path.join(args.viz_dir, 'step_*.png')))
            stride = max(1, int(args.vid_stride))
            frames = frames[::stride]
            if frames:
                first = cv2.imdecode(np.fromfile(frames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
                h, w = first.shape[:2]
                out_path = args.video_out if args.video_out else os.path.join(args.viz_dir, 'nav_run.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vw = cv2.VideoWriter(out_path, fourcc, float(args.fps), (w, h))
                for fpath in frames:
                    img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if img.shape[1] != w or img.shape[0] != h:
                        img = cv2.resize(img, (w, h))
                    vw.write(img)
                vw.release()
                print(f"[video] saved to: {out_path}")
        except Exception as e:
            print(f"[video] failed: {e}")


if __name__ == "__main__":
    main()
