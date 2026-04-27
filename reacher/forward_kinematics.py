import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):

  axis = np.asarray(axis, dtype=float).reshape(3)
  axis = axis / np.linalg.norm(axis)
  x, y, z = axis
  K = np.array([
    [0.0, -z,  y],
    [z,   0.0, -x],
    [-y,  x,   0.0],
  ])
  rot_mat = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  R = rotation_matrix(axis, angle)
  v_A = np.asarray(v_A, dtype=float).reshape(3, 1)
  T = np.block([
    [R, v_A],
    [np.array([[0.0, 0.0, 0.0, 1.0]])]
  ])
  return T

def fk_hip(joint_angles):
  hip_angle = joint_angles[0]
  hip_frame = homogenous_transformation_matrix(
    axis=np.array([0.0, 0.0, 1.0]),
    angle=hip_angle,
    v_A=np.array([0.0, 0.0, 0.0]),
  )
  return hip_frame

def fk_shoulder(joint_angles):
  shoulder_angle = joint_angles[1]
  hip_frame = fk_hip(joint_angles)
  T_hip_to_shoulder = homogenous_transformation_matrix(
    axis=np.array([0.0, 1.0, 0.0]),
    angle=shoulder_angle,
    v_A=np.array([0.0, -HIP_OFFSET, 0.0]),
  )
  shoulder_frame = hip_frame @ T_hip_to_shoulder
  return shoulder_frame

def fk_elbow(joint_angles):
  elbow_angle = joint_angles[2]
  shoulder_frame = fk_shoulder(joint_angles)
  T_shoulder_to_elbow = homogenous_transformation_matrix(
    axis=np.array([0.0, 1.0, 0.0]),
    angle=elbow_angle,
    v_A=np.array([0.0, 0.0, UPPER_LEG_OFFSET]),
  )
  elbow_frame = shoulder_frame @ T_shoulder_to_elbow
  return elbow_frame

def fk_foot(joint_angles):
  
  elbow_frame = fk_elbow(joint_angles)
  T_elbow_to_foot = homogenous_transformation_matrix(
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    v_A=np.array([0.0, 0.0, LOWER_LEG_OFFSET]),
  )
  end_effector_frame = elbow_frame @ T_elbow_to_foot
  return end_effector_frame