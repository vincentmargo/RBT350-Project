import time

import pybullet as p
import numpy as np
np.set_printoptions(suppress=True, precision=3)

from absl import app
from absl import flags

# Support both invocation styles:
# 1) python -m reacher.reacher_manual_control
# 2) python reacher/reacher_manual_control.py
if __package__:
	from . import forward_kinematics
	from . import inverse_kinematics
	from . import reacher_sim_utils
	from .dynamixel_interface import Reacher
else:
	import forward_kinematics
	import inverse_kinematics
	import reacher_sim_utils
	from dynamixel_interface import Reacher


flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
flags.DEFINE_bool("ik"          , False, "Whether to control arms through cartesian coordinates(IK) or joint angles")
flags.DEFINE_list("set_joint_positions", [], "List of joint angles to set at initialization.")
flags.DEFINE_list(
    "set_joint_angles",
    [],
    "Same as set_joint_positions (Hands-on PDF uses this name). If both are set, set_joint_positions wins.",
)
flags.DEFINE_bool("real_to_sim", False, "Whether we want to command the sim robot by moving the real robot")
flags.DEFINE_bool("sim_to_real", False, "Whether we want to command the real robot by moving the sim robot (by moving the slider)")
flags.DEFINE_integer(
    "max_ik_iterations",
    10,
    "Max Newton iterations per IK solve (Problem 13: try 10 vs 1; Problem 14: use 100).",
)
flags.DEFINE_float(
    "ik_fk_tolerance",
    0.05,
    "Max |FK_foot - target| (m) to accept IK when --run_on_robot (raise slightly if FK vs PyBullet mismatch).",
)
flags.DEFINE_float(
	"ik_max_jacobian_cond",
	1e4,
	"Max allowable condition number for numerical Jacobian before we freeze (avoids singularity twitching).",
)
flags.DEFINE_float(
	"ik_max_step",
	0.5,
	"Max joint change (rad) applied per control cycle (clamps sudden IK jumps).",
)
flags.DEFINE_float(
	"ik_smoothing",
	0.0,
	"Exponential smoothing factor for IK commands in [0,1). 0 = no smoothing, 0.8 = very smooth.",
)
flags.DEFINE_integer(
	"ik_stall_iters",
	25,
	"Freeze IK if FK error fails to improve for this many cycles.",
)
flags.DEFINE_float(
	"ik_min_improve",
	1e-4,
	"Minimum FK error improvement per cycle to reset stall counter (meters).",
)
flags.DEFINE_float("ws_min_x", -0.2, "Workspace min X (m) for IK target clamping.")
flags.DEFINE_float("ws_max_x", 0.2, "Workspace max X (m) for IK target clamping.")
flags.DEFINE_float("ws_min_y", -0.2, "Workspace min Y (m) for IK target clamping.")
flags.DEFINE_float("ws_max_y", 0.2, "Workspace max Y (m) for IK target clamping.")
flags.DEFINE_float("ws_min_z", -0.08, "Workspace min Z (m) for IK target clamping.")
flags.DEFINE_float("ws_max_z", 0.2, "Workspace max Z (m) for IK target clamping.")
flags.DEFINE_float("ws_cyl_min_r", 0.04, "Workspace cylindrical inner radius sqrt(x^2+y^2) (m).")
flags.DEFINE_float("ws_cyl_max_r", 0.2, "Workspace cylindrical max radius sqrt(x^2+y^2) (m) for IK target clamping.")
FLAGS = flags.FLAGS

UPDATE_DT = 0.01  # seconds


def _initial_joint_list():
    """PDF Problem 14 uses --set_joint_angles; repo originally used --set_joint_positions."""
    if FLAGS.set_joint_positions:
        return FLAGS.set_joint_positions
    if FLAGS.set_joint_angles:
        return FLAGS.set_joint_angles
    return []


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
	"""Clamp xyz to the configured box limits + cylinder radius limit."""
	xyz = np.asarray(xyz, dtype=float).reshape(3)
	x = float(np.clip(xyz[0], FLAGS.ws_min_x, FLAGS.ws_max_x))
	y = float(np.clip(xyz[1], FLAGS.ws_min_y, FLAGS.ws_max_y))
	z = float(np.clip(xyz[2], FLAGS.ws_min_z, FLAGS.ws_max_z))

	# Cylindrical clamp about the z-axis: ws_cyl_min_r <= r <= ws_cyl_max_r
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


def main(argv):
	run_on_robot = FLAGS.run_on_robot
	reacher_sim = reacher_sim_utils.load_reacher()

	# Sphere markers for the students' FK solutions
	shoulder_sphere_id = reacher_sim_utils.create_debug_sphere([1, 0, 0, 1])
	elbow_sphere_id    = reacher_sim_utils.create_debug_sphere([0, 1, 0, 1])
	foot_sphere_id     = reacher_sim_utils.create_debug_sphere([0, 0, 1, 1])
	target_sphere_id   = reacher_sim_utils.create_debug_sphere([1, 1, 1, 1], radius=0.01)

	joint_ids = reacher_sim_utils.get_joint_ids(reacher_sim)
	param_ids = reacher_sim_utils.get_param_ids(reacher_sim, FLAGS.ik)
	reacher_sim_utils.zero_damping(reacher_sim)

	p.setPhysicsEngineParameter(numSolverIterations=10)

	# Set up physical robot if we're using it
	sim_to_real = False
	real_to_sim = False
	if run_on_robot:
		reacher_real = Reacher()
		time.sleep(0.25)
		real_to_sim = FLAGS.real_to_sim
		sim_to_real = FLAGS.sim_to_real
		# reacher_real.reset()

	# Control Loop Variables
	p.setRealTimeSimulation(1)
	counter = 0
	last_command = time.time()
	joint_positions = np.zeros(3)
	halt_real_robot = False  # per-cycle gate for real-robot commanding
	last_safe_joint_positions = None
	last_commanded_joint_positions = None
	last_fk_err = None
	stall_count = 0
	last_valid_xyz = None

	if _initial_joint_list():
		# First set the joint angles to 0,0,0
		for idx, joint_id in enumerate(joint_ids):
			p.setJointMotorControl2(
			reacher_sim,
			joint_id,
			p.POSITION_CONTROL,
			joint_positions[idx],
			force=2.
			)
		joint_positions = np.array(_initial_joint_list(), dtype=np.float32)
		# Set the simulated robot to match the joint angles
		for idx, joint_id in enumerate(joint_ids):
			p.setJointMotorControl2(
			reacher_sim,
			joint_id,
			p.POSITION_CONTROL,
			joint_positions[idx],
			force=2.
			)


	# Main loop
	while (True):

		# Control loop
		if time.time() - last_command > UPDATE_DT:
			last_command = time.time()
			counter += 1

			# Read the slider values
			try:
				slider_values = np.array([p.readUserDebugParameter(id) for id in param_ids])
			except:
				pass
			if FLAGS.ik:
				# Always display the raw slider target (even if invalid).
				raw_xyz = np.asarray(slider_values, dtype=float).reshape(3)
				p.resetBasePositionAndOrientation(target_sphere_id, posObj=raw_xyz, ornObj=[0, 0, 0, 1])

				# If the user drags the target outside the workspace, hold the last valid target.
				if last_valid_xyz is None:
					last_valid_xyz = _clamp_workspace_xyz(slider_values)
				if _workspace_contains_xyz(slider_values):
					last_valid_xyz = np.asarray(slider_values, dtype=float).reshape(3)
				# IK uses last valid target.
				xyz = last_valid_xyz.copy()
			else:
				sim_target_joint_positions = slider_values

			# Read the simulated robot's joint angles
			sim_joint_positions = []
			for idx, joint_id in enumerate(joint_ids):
				sim_joint_positions.append(p.getJointState(reacher_sim, joint_id)[0])
			sim_joint_positions = np.array(sim_joint_positions)
			if last_safe_joint_positions is None:
				last_safe_joint_positions = sim_joint_positions[:3].copy()
			if last_commanded_joint_positions is None:
				last_commanded_joint_positions = sim_joint_positions[:3].copy()
			
			
			# If IK is enabled, update joint angles based off of goal XYZ position
			if FLAGS.ik:
				# Reset per-cycle safety gate; we'll re-enable it only if IK is invalid.
				halt_real_robot = False
				ik_valid = True
				ret = inverse_kinematics.calculate_inverse_kinematics(
					xyz,
					sim_joint_positions[:3],
					max_iterations=FLAGS.max_ik_iterations,
				)
				if ret is not None:
					enable = True
					# Wraps angles between -pi, pi
					ik_joint_positions = np.arctan2(np.sin(ret), np.cos(ret))

					# Double check that the angles are a correct solution before sending anything to the real robot
					# If the error between the goal foot position and the position of the foot obtained from the IK solution is too large,
					# don't set the joint angles of the robot to the angles obtained from IK 
					pos = forward_kinematics.fk_foot(ik_joint_positions[:3])[:3, 3]
					fk_err = float(np.linalg.norm(np.asarray(pos) - xyz))

					# Jacobian conditioning check (numerical stability / singularities).
					J = inverse_kinematics.calculate_jacobian_FD(
						ik_joint_positions[:3], inverse_kinematics.PERTURBATION
					)
					try:
						j_cond = float(np.linalg.cond(J))
					except Exception:
						j_cond = float("inf")

					# Stall detection: if error isn't improving, freeze.
					if last_fk_err is not None and (last_fk_err - fk_err) < FLAGS.ik_min_improve:
						stall_count += 1
					else:
						stall_count = 0
					last_fk_err = fk_err

					# Validity gates
					if fk_err > FLAGS.ik_fk_tolerance:
						ik_valid = False
					if not np.isfinite(j_cond) or j_cond > FLAGS.ik_max_jacobian_cond:
						ik_valid = False
					if stall_count >= FLAGS.ik_stall_iters:
						ik_valid = False

					if FLAGS.run_on_robot and not ik_valid:
						halt_real_robot = True
						print("Prevented operation on real robot as inverse kinematics solution was not correct")

					if ik_valid:
						# Step limiting in joint space (prevents snapping)
						dq = ik_joint_positions[:3] - sim_joint_positions[:3]
						dq_norm = float(np.linalg.norm(dq))
						if dq_norm > FLAGS.ik_max_step and dq_norm > 1e-9:
							dq = dq * (FLAGS.ik_max_step / dq_norm)
						sim_target_joint_positions = sim_joint_positions[:3] + dq

						# Optional smoothing
						alpha = float(FLAGS.ik_smoothing)
						if alpha > 0.0:
							sim_target_joint_positions = (
								alpha * last_commanded_joint_positions + (1.0 - alpha) * sim_target_joint_positions
							)

						last_commanded_joint_positions = sim_target_joint_positions.copy()
						last_safe_joint_positions = sim_target_joint_positions.copy()
					else:
						# Freeze at last safe command (stops twitching on unreachable targets)
						sim_target_joint_positions = last_safe_joint_positions.copy()
				else:
					# If IK failed to return a value, freeze.
					sim_target_joint_positions = last_safe_joint_positions.copy()
					if FLAGS.run_on_robot:
						halt_real_robot = True
						print("Prevented operation on real robot as inverse kinematics solution was not correct")
			else:
				# Non-IK mode: keep last_safe in sync with slider target.
				last_safe_joint_positions = sim_target_joint_positions[:3].copy()

			# If real-to-sim, update the simulated robot's joint angles based on the real robot's joint angles
			if real_to_sim:
				sim_target_joint_positions = reacher_real.get_joint_positions()
				sim_target_joint_positions[1] *= -1

			# Set the simulated robot's joint positions to sim_target_joint_positions
			for idx, joint_id in enumerate(joint_ids):
				p.setJointMotorControl2(
					reacher_sim,
					joint_id,
					p.POSITION_CONTROL,
					sim_target_joint_positions[idx],
					force=2.
				)

			# If sim-to-real, update the real robot's joint angles based on the simulated robot's joint angle
			if sim_to_real and not halt_real_robot:
				target_joint_positions = sim_joint_positions.copy()
				target_joint_positions[1] *= -1
				reacher_real.set_joint_positions(target_joint_positions)
			
			
			# Obtain the real robot's joint angles
			if run_on_robot:
				real_joint_positions = reacher_real.get_joint_positions()


			# Get the calculated positions of each joint and the end effector
			shoulder_pos = forward_kinematics.fk_shoulder(sim_joint_positions[:3])[:3,3]
			elbow_pos    = forward_kinematics.fk_elbow(sim_joint_positions[:3])[:3,3]
			foot_pos     = forward_kinematics.fk_foot(sim_joint_positions[:3])[:3,3]

			# Show the debug spheres for FK
			p.resetBasePositionAndOrientation(shoulder_sphere_id, posObj=shoulder_pos, ornObj=[0, 0, 0, 1])
			p.resetBasePositionAndOrientation(elbow_sphere_id   , posObj=elbow_pos   , ornObj=[0, 0, 0, 1])
			p.resetBasePositionAndOrientation(foot_sphere_id    , posObj=foot_pos    , ornObj=[0, 0, 0, 1])

			# This is a small hack. Ignore this.
			if _initial_joint_list() and counter == 1:
				time.sleep(2)

			# Show the result in the terminal
			if counter % 20 == 0:
				print("Simulated robot joint positions: ", sim_joint_positions)
				if run_on_robot:
					print("Real robot joint positions: ", reacher_real.get_joint_positions())

app.run(main)
