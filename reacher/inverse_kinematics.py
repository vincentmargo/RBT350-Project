import math
import numpy as np
import copy
from reacher import forward_kinematics

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2
TOLERANCE = 0.01 # tolerance for inverse kinematics
PERTURBATION = 0.0001 # perturbation for finite difference method
MAX_ITERATIONS = 10

"""
Example invocation:

python -m reacher.reacher_manual_control --ik --run_on_robot --sim_to_real \\
  --max_ik_iterations=30 --ik_fk_tolerance=0.05 --ik_max_step=0.7 --ik_smoothing=0.2 \\
  --set_joint_angles=0,0.1,0.1
"""

def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics cost.

    This function computes the inverse kinematics cost, which represents the Euclidean
    distance between the desired end-effector position and the end-effector position
    resulting from the provided 'guess' joint angles.

    Args:
        end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
            position. A numpy array with 3 elements.

    Returns:
        float: The Euclidean distance between end_effector_pos and the calculated end-effector
        position based on the guess.
    """
    end_effector_pos = np.asarray(end_effector_pos, dtype=float).reshape(3)
    guess = np.asarray(guess, dtype=float).reshape(3)
    current_pos = forward_kinematics.fk_foot(guess)[:3, 3]
    cost = float(np.linalg.norm(end_effector_pos - current_pos))
    return cost

def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix using finite differences.

    This function computes the Jacobian matrix for a given set of joint angles using finite differences.

    Args:
        joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
        delta (float): The perturbation value used to approximate the partial derivatives.

    Returns:
        numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
        between joint velocity and end-effector linear velocity.
    """

    joint_angles = np.asarray(joint_angles, dtype=float).reshape(3)
    p0 = forward_kinematics.fk_foot(joint_angles)[:3, 3]
    J = np.zeros((3, 3))
    for j in range(3):
        q_pert = joint_angles.copy()
        q_pert[j] += delta
        p_pert = forward_kinematics.fk_foot(q_pert)[:3, 3]
        J[:, j] = (p_pert - p0) / delta
    return J

def calculate_inverse_kinematics(end_effector_pos, guess, max_iterations=None):
    """
    Calculate the inverse kinematics solution using the Newton-Raphson method.

    This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
    It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

    Args:
        end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.
        max_iterations (int, optional): Override for MAX_ITERATIONS (e.g. Problem 13: 10 vs 1).

    Returns:
        numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
    """

    end_effector_pos = np.asarray(end_effector_pos, dtype=float).reshape(3)
    guess = np.asarray(guess, dtype=float).reshape(3).copy()
    previous_cost = np.inf
    n_iters = MAX_ITERATIONS if max_iterations is None else int(max_iterations)

    for iters in range(n_iters):
        J = calculate_jacobian_FD(guess, PERTURBATION)
        current_pos = forward_kinematics.fk_foot(guess)[:3, 3]
        residual = end_effector_pos - current_pos
        dq = np.linalg.pinv(J) @ residual
        guess = guess + dq
        cost = ik_cost(end_effector_pos, guess)
        if abs(previous_cost - cost) < TOLERANCE:
            break
        previous_cost = cost

    return guess
