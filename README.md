# pepper_kinematics

## Description 

Originally, Pepper does not provide any inverse kinematics function. This provides simple inverse kinematics funciton.

## Example

    """
    Move work_pose first, then move 5 centimeters toward left (y axis positive side).
    """
    
    import time

    from naoqi import ALProxy
    import pepper_kinematics as pk

    import numpy as np

    host = '127.0.0.1'
    port = 9559

    m = ALProxy("ALMotion", host, port)
    m.setAngles(pk.left_arm_tags, pk.left_arm_work_pose, 1.0)

    print pk.left_arm_work_pose[1]
    print "--------------------------"

    time.sleep(1.0)

    current_angles = m.getAngles(pk.left_arm_tags, True)
    current_position, current_orientation = pk.left_arm_get_position(current_angles)

    target_position = current_position
    target_position[1] = target_position[1] + 0.05 # 5 cm toward left
    target_orientation = current_orientation # This is not supported yet

    target_angles = pk.left_arm_set_position(current_angles, target_position, target_orientation)
    print target_angles.tolist()
    if target_angles.any():
        m.setAngles(pk.left_arm_tags, target_angles.tolist(), 1.0)


## How to install
    git clone https://github.com/m0rph03nix/python_pepper_kinematics.git
    cd python_pepper_kinematics
    python setup.py install --user

## Copyright
* author: Yuki Suga
* copyright: Yuki Suga @ ssr.tokyo
* license: GPLv3

