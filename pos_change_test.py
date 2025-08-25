from lib.navigation.icp_nav_utils import (
    R_S_to_C,
    R_C_to_S,
    convert_R_C_to_S,
    fuse_icp_delta_into_pitch_yaw,
    extract_yaw_pitch_from_R_S,
)

import pybullet as p
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def deg(x):
    return np.degrees(x)

def rad(x):
    return np.radians(x)

def main():
    # 假设当前姿态 (S 系)
    pitch_deg, yaw_deg = 10.0, -5.0
    quat = p.getQuaternionFromEuler([rad(pitch_deg), 0.0, rad(yaw_deg)])
    R_S = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    print('[S] R_S=\n', R_S)

    # 1) S↔C 往返验证
    R_C = R_S_to_C @ R_S @ R_C_to_S
    R_S_rec = R_C_to_S @ R_C @ R_S_to_C
    err_roundtrip = np.linalg.norm(R_S - R_S_rec)
    print('[S<->C] round-trip error:', err_roundtrip)

    # 2) 从 R_S 提取回 pitch,yaw
    pitch_rec, yaw_rec = extract_yaw_pitch_from_R_S(R_S)
    print('[extract] pitch,yaw rec= (%.4f, %.4f)' % (pitch_rec, yaw_rec))

    # 3) 构造一个期望的 S 系小增量，并转到 C 系作为“ICP输出”
    d_pitch_sim, d_yaw_sim = -0.8, 1.2  # 期望校正方向
    quat_delta_S = p.getQuaternionFromEuler([rad(d_pitch_sim), 0.0, rad(d_yaw_sim)])
    R_delta_S = np.array(p.getMatrixFromQuaternion(quat_delta_S)).reshape(3, 3)
    R_delta_C = R_S_to_C @ R_delta_S @ R_C_to_S

    # 在 S 系直接提取增量，作为真值对照
    dpitch_S_true, dyaw_S_true = extract_yaw_pitch_from_R_S(R_delta_S)
    print('[delta S true] dpitch,dyaw= (%.4f, %.4f)' % (dpitch_S_true, dyaw_S_true))

    # 把 C 系增量转回 S 系并提取，验证一致性
    R_delta_S_fromC = convert_R_C_to_S(R_delta_C)
    dpitch_S_fromC, dyaw_S_fromC = extract_yaw_pitch_from_R_S(R_delta_S_fromC)
    print('[delta C->S] dpitch,dyaw= (%.4f, %.4f)' % (dpitch_S_fromC, dyaw_S_fromC))

    # 4) 融合 (rmse 好的情况)
    new_pitch, new_yaw = fuse_icp_delta_into_pitch_yaw(
        pitch_deg, yaw_deg, R_delta_C, alpha=0.5, rmse=0.005
    )
    print('[fuse ok] old: (%.4f, %.4f) -> new: (%.4f, %.4f)'
          % (pitch_deg, yaw_deg, new_pitch, new_yaw))

    # 5) 融合 (rmse 差的情况，应该不更新)
    new_pitch_bad, new_yaw_bad = fuse_icp_delta_into_pitch_yaw(
        pitch_deg, yaw_deg, R_delta_C, alpha=0.5, rmse=0.02
    )
    print('[fuse bad] old: (%.4f, %.4f) -> new: (%.4f, %.4f)'
          % (pitch_deg, yaw_deg, new_pitch_bad, new_yaw_bad))

if __name__ == '__main__':
    main()