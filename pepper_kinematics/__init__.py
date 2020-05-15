import numpy as np
import scipy as sp
from scipy import linalg

import forward_kinematics as fk
import inverse_kinematics as ik

right_arm_tags = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
right_arm_initial_pose = [1.0, -0.2, 1.57-0.2, 1.0, -1.57]
right_arm_work_pose = [0.8, -0.2, 1.57-0.2, 0.9, -1.57]

_inverse_case = [1.0, -1.0, -1.0, -1.0, -1.0]

left_arm_tags = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
left_arm_initial_pose = [p[0] * p[1] for p in zip(right_arm_initial_pose, _inverse_case)]
left_arm_work_pose = [p[0] * p[1] for p in zip(right_arm_work_pose, _inverse_case)]


#https://digitalcommons.wpi.edu/cgi/viewcontent.cgi?article=3252&context=mqp-all


#Import right/left_arm_current_pose or calculate it from functions below

def body_limits(right_arm_tags, right_arm_current_pose, left_arm_tags, left_arm_current_pose):
    lower_limits = [-2.0857 -1.5620, -2.0857, 0.0087, -1.8239]
    upper_limits = [2.0857, -0.0087, 2.0857, 1.5620, 1.8239]
    step = 0
    message = ""
    while ((message == "") and (step =< 4)):
        if ((lower_limits[step] < right_arm_current_pose[step]) or (upper_limits[step] > right_arm_current_pose[step]):
            message = "Error, position inaccurate")
        step += 1
    print(message)


def Stand(R_target_pos, R_target_ori, L_target_pos, L_target_ori):
    R_angles = [1.57, 0.0087, -1.57, -0.0087, 0.0]
    L_angles = [1.57, -1.57, -1.0, -0.0087, 0.0]
    right_arm_set_position(R_angles, R_target_pos, R_target_ori, R_epsilon=0.0001)
    left_arm_set_position(L_angles, L_target_pos, L_target_ori, L_epsilon = 0.0001)

def StandZero(R_target_pos, R_target_ori, L_target_pos, L_target_ori):
    R_angles = [0.0, 0.0087, 0.0, -0.0087, 0.0]
    L_angles = [0.0, 0.0, -1.0, -0.0087, 0.0]    
    right_arm_set_position(R_angles, R_target_pos, R_target_ori, R_epsilon=0.0001)
    left_arm_set_position(L_angles, L_target_pos, L_target_ori, L_epsilon = 0.0001)

def Crouch(R_target_pos, R_target_ori L_target_pos, L_target_ori):
    R_angles = [1.57, 0.0087, -1.8239, -0.0087, 0.0]
    L_angles = [1.57, -1.8239, -1.0, -0.0087, 0.0]    
    right_arm_set_position(R_angles, R_target_pos, R_target_ori, R_epsilon=0.0001)
    left_arm_set_position(L_angles, L_target_pos, L_target_ori, L_epsilon = 0.0001)




    
def right_arm_get_position(angles):
    """
    Just calculate the position when joints on the pepper's right arm is in given positions

    Args:
      angles : Angles of right arm joints (list of 5 double values. unit is radian)
    
    Returns:
      A tuple of two arrays (position, orientation). orientation is presented as Matrix. Unit = meter.
      
      (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """
    return fk.calc_fk_and_jacob(angles, jacob=False, right=True)

def left_arm_get_position(angles):
    """
    Just calculate the position when joints on the pepper's left arm is in given positions

    Args:
      angles : Angles of left arm joints (list of 5 double values. unit is radian)
    
    Returns:
      A tuple of two arrays (position, orientation). orientation is presented as Matrix. Unit = meter.
      
      (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """
    return fk.calc_fk_and_jacob(angles, jacob=False, right=False)

def right_arm_set_position(angles, target_pos, target_ori, epsilon=0.0001):
    """
    Just calculate the joint angles when the Pepper's right hand position is in the given position
    
    Args:
      angles : Use the initial position of calculation. Unit = radian
      target_pos : List. [Px, Py, Pz]. Unit is meter.
      target_ori : np.array([[R00,R01,R02],[R10,R11,R12],[R20,R21,R22]])
      epsilon    : The threshold. If the distance between calculation result and target_position is lower than epsilon, this returns value.
    
    Returns:
      A list of joint angles (Unit is radian). If calculation fails, return None.
    """
    return ik.calc_inv_pos(angles, target_pos, target_ori, epsilon, right=True)

def left_arm_set_position(angles, target_pos, target_ori, epsilon = 0.0001):
    return ik.calc_inv_pos(angles, target_pos, target_ori, epsilon, right=False)

def right_orientation(angles):
    """
    (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """
    (position, orientation) = right_arm_get_position(angles)
    orientation[2][0] = - (orientation[0][0] + orientation[1][0])
    orientation[2][1] = - (orientation[0][1] + orientation[1][1])
    orientation[2][2] = - (orientation[0][2] + orientation[1][2])
    return orientation

def left_orientation(angles):
    """
    (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """
    (position, orientation) = left_arm_get_position(angles)
    orientation[2][0] = - (orientation[0][0] + orientation[1][0])
    orientation[2][1] = - (orientation[0][1] + orientation[1][1])
    orientation[2][2] = - (orientation[0][2] + orientation[1][2])
    return orientation
    
def left_body_limits (angles, epsilon=0.001, left_asked_position):
    """
    (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """
    (left_position, left_orientation) = left_arm_get_position(angles)
    res_temp = False
    res = "Within limits"
    step = 0
    delta = 0
    while (res == False and delta < epsilon): 
        delta = left_asked_position[0] - left_position[0]
        if(delta <= epsilon):
           res = True
        left_position[0] -= step
        step += 0.0001
        left_position[0] += step
   
    if res_temp == True
        res_temp = False
    else:
        res = "Beyond limits"
        
        
    while (res == False and delta < epsilon): 
        delta = left_asked_position[0] - left_position[0]
        if(delta <= epsilon):
           res = True
        left_position[0] -= step
        step += 0.0001
        left_position[0] += step
        
def right_body_limits (angles, epsilon=0.001, right_asked_position, right_asked_orientation):
    """
    (position, orientation) = (np.array([position_x, position_y, position_z]), np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]]))
    """ 
    (right_position, right_orientation) = right_arm_get_position(angles)
    res = False
    while (res == False and step < epsilon): 
        if(right_asked_position[0] - right_position[0]) <= epsilon):
           res = True
    
    
