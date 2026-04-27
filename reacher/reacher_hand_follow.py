import time
from typing import Optional

import numpy as np
np.set_printoptions(suppress=True, precision=3)

import pybullet as p
from absl import app
from absl import flags


# Support both invocation styles:
# 1) python -m reacher.reacher_hand_follow
# 2) python reacher/reacher_hand_follow.py
if __package__:
    from . import forward_kinematics
    from . import inverse_kinematics
    from . import reacher_sim_utils
    from .vision_hand_tracker import HandTracker, RedDotTracker
    from .dynamixel_interface import Reacher
else:
    import forward_kinematics
    import inverse_kinematics
    import reacher_sim_utils
    from vision_hand_tracker import HandTracker, RedDotTracker
    from dynamixel_interface import Reacher


flags.DEFINE_bool("run_on_robot", False, "If true, connect to Dynamixel arm.")
flags.DEFINE_bool("sim_to_real", False, "If true, command real arm from simulated joints.")

flags.DEFINE_integer("camera_index", 0, "OpenCV camera index (usually 0).")
flags.DEFINE_integer("camera_width", 640, "Requested camera capture width.")
flags.DEFINE_integer("camera_height", 480, "Requested camera capture height.")
flags.DEFINE_integer("camera_fps", 30, "Requested camera FPS (best-effort).")
flags.DEFINE_string(
    "camera_fourcc",
    "MJPG",
    "Requested camera pixel format fourcc (e.g. MJPG, YUYV). Best-effort.",
)
flags.DEFINE_bool("mirror", True, "Mirror the camera feed (more intuitive for users).")
flags.DEFINE_bool("show_vision", True, "Show a separate OpenCV window with overlays.")
flags.DEFINE_string(
    "hand_point_mode",
    "palm",
    "Hand point to track: palm (stable) or index_tip (more responsive).",
)
flags.DEFINE_float(
    "min_hand_score",
    0.02,
    "Minimum detection score to accept a hand point (mainly affects skin-fallback).",
)
flags.DEFINE_string(
    "track_mode",
    "red_dot",
    "What to track from the camera: hand or red_dot.",
)
flags.DEFINE_float(
    "min_dot_area_px",
    80.0,
    "Minimum red-dot contour area (pixels) to accept a detection.",
)
flags.DEFINE_float(
    "dot_max_area_frac",
    0.10,
    "Maximum dot area as a fraction of image area (rejects giant false blobs).",
)
flags.DEFINE_float(
    "dot_min_circularity",
    0.75,
    "Minimum contour circularity (1.0 = perfect circle). Helps reject non-dot blobs.",
)
flags.DEFINE_integer(
    "dot_dark_threshold",
    -1,
    "Fallback detector threshold in [0..255] for dark blobs; -1 uses Otsu.",
)
flags.DEFINE_integer(
    "camera_autoscan_max",
    6,
    "If the requested camera_index fails, scan indices [0..N-1] to find an available camera.",
)
flags.DEFINE_bool(
    "require_camera",
    True,
    "If false, allow running with no webcam (holds pose) instead of raising an exception.",
)

flags.DEFINE_float("x0", 0.00, "Workspace mapping center X (m) in IK/FK base frame.")
flags.DEFINE_float("y0", 0.10, "Workspace mapping center Y (m) in IK/FK base frame.")
flags.DEFINE_float("x_range", 0.4, "Workspace mapping X span (m) across the image width.")
flags.DEFINE_float("y_range", 0.24, "Workspace mapping Y span (m) across the image height.")
flags.DEFINE_float("z_fixed", 0.05, "Fixed Z target (m) in IK/FK base frame.")
flags.DEFINE_bool(
    "use_z_from_v",
    True,
    "If true, map vertical image motion (v) to Z instead of using z_fixed.",
)
flags.DEFINE_float("z0", 0.06, "Workspace mapping center Z (m) when use_z_from_v is enabled.")
flags.DEFINE_float("z_range", 0.28, "Workspace mapping Z span (m) across the image height when use_z_from_v is enabled.")
flags.DEFINE_string(
    "fixed_axis",
    "y",
    "Which robot axis to keep fixed (one of: x, y, z). The other two axes are controlled by (u,v).",
)
flags.DEFINE_float("x_fixed", 0.00, "Fixed X target (m) when fixed_axis=x.")
flags.DEFINE_float("y_fixed", 0.10, "Fixed Y target (m) when fixed_axis=y.")
flags.DEFINE_float("z_fixed2", 0.05, "Fixed Z target (m) when fixed_axis=z (preferred over z_fixed).")

# 2x2 linear mapping from (du,dv) to the two free axes (lets you rotate camera axes to robot axes).
# Default is diagonal scaling: a = a0 + du*x_range, b = b0 + dv*y_range (with dv inverted in code).
flags.DEFINE_float("uv_m00", 1.0, "2x2 mapping matrix element m00.")
flags.DEFINE_float("uv_m01", 0.0, "2x2 mapping matrix element m01.")
flags.DEFINE_float("uv_m10", 0.0, "2x2 mapping matrix element m10.")
flags.DEFINE_float("uv_m11", 1.0, "2x2 mapping matrix element m11.")

flags.DEFINE_bool(
    "vision_mask_only",
    True,
    "If true (and track_mode=red_dot), display a black/white mask of the detected pixels instead of the full image.",
)

flags.DEFINE_float("ema_alpha", 0.75, "EMA smoothing alpha in [0,1). Higher = smoother/slower.")
flags.DEFINE_integer("lost_hand_frames", 15, "If no hand for N frames, hold pose and stop real commanding.")
flags.DEFINE_string(
    "lost_behavior",
    "home",
    "Behavior when target is lost: hold or home.",
)
flags.DEFINE_list(
    "home_joint_positions",
    ["0.0", "0.1", "0.1"],
    "Home joint positions (rad) used when lost_behavior=home.",
)
flags.DEFINE_float("target_rate_hz", 60.0, "Control loop update rate (Hz).")

flags.DEFINE_integer("max_ik_iterations", 10, "Max Newton iterations per IK solve.")
flags.DEFINE_float("ik_fk_tolerance", 0.05, "Max |FK_foot - target| (m) to accept IK.")
flags.DEFINE_float("ik_max_jacobian_cond", 1e4, "Max allowable Jacobian condition number.")
flags.DEFINE_float("ik_max_step", 0.5, "Max joint change norm (rad) applied per cycle.")
flags.DEFINE_float("ik_smoothing", 0.0, "Extra joint-space smoothing factor in [0,1).")

flags.DEFINE_float("ws_min_x", -0.2, "Workspace min X (m) for IK target clamping.")
flags.DEFINE_float("ws_max_x", 0.2, "Workspace max X (m) for IK target clamping.")
flags.DEFINE_float("ws_min_y", -0.2, "Workspace min Y (m) for IK target clamping.")
flags.DEFINE_float("ws_max_y", 0.2, "Workspace max Y (m) for IK target clamping.")
flags.DEFINE_float("ws_min_z", -0.08, "Workspace min Z (m) for IK target clamping.")
flags.DEFINE_float("ws_max_z", 0.2, "Workspace max Z (m) for IK target clamping.")
flags.DEFINE_float("ws_cyl_min_r", 0.04, "Workspace cylindrical inner radius sqrt(x^2+y^2) (m).")
flags.DEFINE_float("ws_cyl_max_r", 0.2, "Workspace cylindrical max radius sqrt(x^2+y^2) (m).")

FLAGS = flags.FLAGS

def _update_dt() -> float:
    hz = float(FLAGS.target_rate_hz)
    if not np.isfinite(hz) or hz <= 1e-3:
        return 0.01
    return 1.0 / hz


def _workspace_contains_xyz(xyz: np.ndarray) -> bool:
    xyz = np.asarray(xyz, dtype=float).reshape(3)
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    if x < FLAGS.ws_min_x or x > FLAGS.ws_max_x:
        return False
    if y < FLAGS.ws_min_y or y > FLAGS.ws_max_y:
        return False
    if z < FLAGS.ws_min_z or z > FLAGS.ws_max_z:
        return False
    r_max = float(FLAGS.ws_cyl_max_r)
    r_min = float(FLAGS.ws_cyl_min_r)
    r = float(np.hypot(x, y))
    if r_max > 0.0 and r > r_max:
        return False
    if r_min > 0.0 and r < r_min:
        return False
    return True


def _clamp_workspace_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float).reshape(3)
    x = float(np.clip(xyz[0], FLAGS.ws_min_x, FLAGS.ws_max_x))
    y = float(np.clip(xyz[1], FLAGS.ws_min_y, FLAGS.ws_max_y))
    z = float(np.clip(xyz[2], FLAGS.ws_min_z, FLAGS.ws_max_z))

    r = float(np.hypot(x, y))
    r_max = float(FLAGS.ws_cyl_max_r)
    r_min = float(FLAGS.ws_cyl_min_r)
    if r_max > 0.0 and r > r_max:
        scale = r_max / r
        x *= scale
        y *= scale
        r = r_max
    if r_min > 0.0 and r < r_min:
        if r > 1e-12:
            scale = r_min / r
            x *= scale
            y *= scale
        else:
            x = r_min
            y = 0.0
    return np.array([x, y, z], dtype=float)


def _pixel_to_xyz(x_px: int, y_px: int, w: int, h: int) -> np.ndarray:
    # Normalize to [0,1]
    u = float(x_px) / max(1, int(w))
    v = float(y_px) / max(1, int(h))

    # du,dv centered at 0. Positive dv should correspond to moving "up" in image.
    du = float(u - 0.5)
    dv = float(0.5 - v)

    # Apply a 2x2 linear transform so you can align camera axes with robot axes:
    # [da]   [m00 m01] [du]
    # [db] = [m10 m11] [dv]
    m00 = float(FLAGS.uv_m00)
    m01 = float(FLAGS.uv_m01)
    m10 = float(FLAGS.uv_m10)
    m11 = float(FLAGS.uv_m11)
    da = m00 * du + m01 * dv
    db = m10 * du + m11 * dv

    fixed_axis = str(FLAGS.fixed_axis).lower().strip()

    # Choose which plane we are commanding. Two axes move with (u,v), one axis is fixed.
    if fixed_axis == "x":
        x = float(FLAGS.x_fixed)
        y = float(FLAGS.y0 + da * FLAGS.x_range)
        z = float(FLAGS.z0 + db * FLAGS.z_range) if bool(FLAGS.use_z_from_v) else float(FLAGS.z_fixed2)
    elif fixed_axis == "y":
        y = float(FLAGS.y_fixed)
        x = float(FLAGS.x0 + da * FLAGS.x_range)
        z = float(FLAGS.z0 + db * FLAGS.z_range) if bool(FLAGS.use_z_from_v) else float(FLAGS.z_fixed2)
    else:
        # fixed z
        z = float(FLAGS.z_fixed2)
        x = float(FLAGS.x0 + da * FLAGS.x_range)
        y = float(FLAGS.y0 + db * FLAGS.y_range)

    return np.array([x, y, z], dtype=float)


def _open_camera(cv2):
    """
    Try to open the requested camera, else scan a few indices.
    Returns (cap, opened_index) or (None, None) if all fail.
    """
    preferred = int(FLAGS.camera_index)
    # Prefer V4L2 backend when available (helps avoid some auto-probing weirdness).
    backend = getattr(cv2, "CAP_V4L2", 0)

    def try_open(idx: int):
        cap = cv2.VideoCapture(int(idx), backend) if backend else cv2.VideoCapture(int(idx))
        if cap is not None and cap.isOpened():
            # Apply capture settings (best-effort; some backends ignore these).
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(FLAGS.camera_width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(FLAGS.camera_height))
            if int(FLAGS.camera_fps) > 0:
                cap.set(cv2.CAP_PROP_FPS, int(FLAGS.camera_fps))
            fmt = str(FLAGS.camera_fourcc or "").upper().strip()
            if len(fmt) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warm up a few reads to reduce first-frame timeouts.
            for _ in range(5):
                cap.read()
            return cap
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        return None

    cap = try_open(preferred)
    if cap is not None:
        return cap, preferred

    max_n = int(max(0, FLAGS.camera_autoscan_max))
    for idx in range(max_n):
        if idx == preferred:
            continue
        cap = try_open(idx)
        if cap is not None:
            return cap, idx

    return None, None


def main(argv):
    import cv2

    reacher_sim = reacher_sim_utils.load_reacher()
    joint_ids = reacher_sim_utils.get_joint_ids(reacher_sim)
    reacher_sim_utils.zero_damping(reacher_sim)
    p.setPhysicsEngineParameter(numSolverIterations=10)
    p.setRealTimeSimulation(1)

    run_on_robot = bool(FLAGS.run_on_robot)
    sim_to_real = bool(FLAGS.sim_to_real)

    halt_real_robot = False
    if run_on_robot:
        reacher_real = Reacher()
        time.sleep(0.25)
    else:
        reacher_real = None

    cap, opened_idx = _open_camera(cv2)
    if cap is None:
        msg = (
            "Could not open any webcam.\n"
            "- On WSL2 this often happens because no /dev/video* devices are exposed.\n"
            "- Try running with Windows Python, or configure webcam passthrough, or try a different --camera_index.\n"
        )
        if bool(FLAGS.require_camera):
            raise RuntimeError(msg)
        print(msg)
    else:
        print(f"Opened webcam index {opened_idx}.")

    track_mode = str(FLAGS.track_mode).lower().strip()
    if track_mode == "red_dot":
        tracker = RedDotTracker(
            mirror=bool(FLAGS.mirror),
            min_area_px=float(FLAGS.min_dot_area_px),
            max_area_frac=float(FLAGS.dot_max_area_frac),
            min_circularity=float(FLAGS.dot_min_circularity),
            dark_threshold=int(FLAGS.dot_dark_threshold),
        )
        min_score = 0.0
    else:
        tracker = HandTracker(mirror=bool(FLAGS.mirror), point_mode=str(FLAGS.hand_point_mode))
        min_score = float(FLAGS.min_hand_score)

    last_command = time.time()
    last_safe_joint_positions = None
    last_commanded_joint_positions = None
    last_fk_err = None
    stall_count = 0

    target_xyz_ema: Optional[np.ndarray] = None
    lost_count = int(FLAGS.lost_hand_frames)
    home_q = np.array([float(x) for x in FLAGS.home_joint_positions], dtype=float).reshape(3)

    # Debug spheres (FK + target)
    shoulder_sphere_id = reacher_sim_utils.create_debug_sphere([1, 0, 0, 1])
    elbow_sphere_id = reacher_sim_utils.create_debug_sphere([0, 1, 0, 1])
    foot_sphere_id = reacher_sim_utils.create_debug_sphere([0, 0, 1, 1])
    target_sphere_id = reacher_sim_utils.create_debug_sphere([1, 1, 1, 1], radius=0.01)

    while True:
        if time.time() - last_command < _update_dt():
            time.sleep(0.001)
            continue
        last_command = time.time()

        if cap is None:
            # No camera available: hold safe pose and skip vision.
            det = None
            fps = None
            disp_frame = None
        else:
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            disp_frame, det, fps = tracker.process_bgr(frame_bgr)
            h, w = disp_frame.shape[0], disp_frame.shape[1]

        # Read sim joints
        sim_joint_positions = np.array([p.getJointState(reacher_sim, jid)[0] for jid in joint_ids], dtype=float)
        if last_safe_joint_positions is None:
            last_safe_joint_positions = sim_joint_positions[:3].copy()
        if last_commanded_joint_positions is None:
            last_commanded_joint_positions = sim_joint_positions[:3].copy()

        status = ""
        halt_real_robot = False

        if det is None or float(getattr(det, "score", 0.0)) < float(min_score):
            lost_count = min(int(FLAGS.lost_hand_frames), lost_count + 1)
            lb = str(FLAGS.lost_behavior).lower().strip()
            if lb == "home":
                status = "NO_TARGET (going_home)"
                # Smoothly move toward home in joint space to avoid twitching.
                dq = home_q - sim_joint_positions[:3]
                dq_norm = float(np.linalg.norm(dq))
                if dq_norm > float(FLAGS.ik_max_step) and dq_norm > 1e-9:
                    dq = dq * (float(FLAGS.ik_max_step) / dq_norm)
                sim_target_joint_positions = sim_joint_positions[:3] + dq
                last_safe_joint_positions = sim_target_joint_positions.copy()
                if run_on_robot:
                    # Don't command real robot when target is lost.
                    halt_real_robot = True
            else:
                status = "NO_TARGET (holding)"
                # Hold last safe pose. Also stop commanding real robot.
                sim_target_joint_positions = last_safe_joint_positions.copy()
                if run_on_robot:
                    halt_real_robot = True
        else:
            lost_count = 0
            raw_xyz = _pixel_to_xyz(det.x_px, det.y_px, w=w, h=h)
            if target_xyz_ema is None:
                target_xyz_ema = raw_xyz.copy()
            alpha = float(np.clip(FLAGS.ema_alpha, 0.0, 0.999))
            target_xyz_ema = alpha * target_xyz_ema + (1.0 - alpha) * raw_xyz

            xyz = _clamp_workspace_xyz(target_xyz_ema)
            if not _workspace_contains_xyz(xyz):
                status = "OUTSIDE_WS (holding)"
                sim_target_joint_positions = last_safe_joint_positions.copy()
                if run_on_robot:
                    halt_real_robot = True
            else:
                # Visualize target
                p.resetBasePositionAndOrientation(target_sphere_id, posObj=xyz, ornObj=[0, 0, 0, 1])

                ret = inverse_kinematics.calculate_inverse_kinematics(
                    xyz, sim_joint_positions[:3], max_iterations=FLAGS.max_ik_iterations
                )
                ik_valid = True
                if ret is None:
                    ik_valid = False
                else:
                    ik_joint_positions = np.arctan2(np.sin(ret), np.cos(ret))
                    pos = forward_kinematics.fk_foot(ik_joint_positions[:3])[:3, 3]
                    fk_err = float(np.linalg.norm(np.asarray(pos) - xyz))

                    J = inverse_kinematics.calculate_jacobian_FD(
                        ik_joint_positions[:3], inverse_kinematics.PERTURBATION
                    )
                    try:
                        j_cond = float(np.linalg.cond(J))
                    except Exception:
                        j_cond = float("inf")

                    if last_fk_err is not None and (last_fk_err - fk_err) < 1e-4:
                        stall_count += 1
                    else:
                        stall_count = 0
                    last_fk_err = fk_err

                    if fk_err > float(FLAGS.ik_fk_tolerance):
                        ik_valid = False
                    if not np.isfinite(j_cond) or j_cond > float(FLAGS.ik_max_jacobian_cond):
                        ik_valid = False
                    if stall_count >= 25:
                        ik_valid = False

                    if ik_valid:
                        dq = ik_joint_positions[:3] - sim_joint_positions[:3]
                        dq_norm = float(np.linalg.norm(dq))
                        if dq_norm > float(FLAGS.ik_max_step) and dq_norm > 1e-9:
                            dq = dq * (float(FLAGS.ik_max_step) / dq_norm)
                        sim_target_joint_positions = sim_joint_positions[:3] + dq

                        joint_alpha = float(np.clip(FLAGS.ik_smoothing, 0.0, 0.999))
                        if joint_alpha > 0.0:
                            sim_target_joint_positions = (
                                joint_alpha * last_commanded_joint_positions
                                + (1.0 - joint_alpha) * sim_target_joint_positions
                            )
                        last_commanded_joint_positions = sim_target_joint_positions.copy()
                        last_safe_joint_positions = sim_target_joint_positions.copy()
                        status = f"OK (fk_err={fk_err:.3f})"
                    else:
                        status = "IK_INVALID (holding)"
                        sim_target_joint_positions = last_safe_joint_positions.copy()
                        if run_on_robot:
                            halt_real_robot = True

        # Command sim
        for idx, joint_id in enumerate(joint_ids):
            p.setJointMotorControl2(
                reacher_sim,
                joint_id,
                p.POSITION_CONTROL,
                float(sim_target_joint_positions[idx]),
                force=2.0,
            )

        # Command real (mirror manual_control sign convention)
        if sim_to_real and run_on_robot and not halt_real_robot and reacher_real is not None:
            target_joint_positions = sim_joint_positions.copy()
            target_joint_positions[1] *= -1
            reacher_real.set_joint_positions(target_joint_positions)

        # FK debug spheres
        shoulder_pos = forward_kinematics.fk_shoulder(sim_joint_positions[:3])[:3, 3]
        elbow_pos = forward_kinematics.fk_elbow(sim_joint_positions[:3])[:3, 3]
        foot_pos = forward_kinematics.fk_foot(sim_joint_positions[:3])[:3, 3]
        p.resetBasePositionAndOrientation(shoulder_sphere_id, posObj=shoulder_pos, ornObj=[0, 0, 0, 1])
        p.resetBasePositionAndOrientation(elbow_sphere_id, posObj=elbow_pos, ornObj=[0, 0, 0, 1])
        p.resetBasePositionAndOrientation(foot_sphere_id, posObj=foot_pos, ornObj=[0, 0, 0, 1])

        if bool(FLAGS.show_vision) and disp_frame is not None:
            vis_frame = disp_frame
            if track_mode == "red_dot" and bool(FLAGS.vision_mask_only):
                mask = getattr(tracker, "get_last_mask_u8", lambda: None)()
                if mask is not None:
                    # Show black/white mask (convert to BGR for text overlay).
                    vis_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            overlay = tracker.draw_overlay(
                disp_frame,
                det,
                target_xyz=target_xyz_ema,
                status=status,
                fps=fps,
            )
            # If we're showing mask-only, keep overlays/text but base it on the mask image.
            if vis_frame is not disp_frame:
                overlay = tracker.draw_overlay(
                    vis_frame,
                    det,
                    target_xyz=target_xyz_ema,
                    status=status,
                    fps=fps,
                )
            cv2.imshow("Robot vision (hand follow)", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


app.run(main)

