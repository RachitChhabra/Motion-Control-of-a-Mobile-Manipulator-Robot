import numpy as np
import modern_robotics as mr
import csv
from scipy.linalg import expm

################### MileStone 1 ########################

##------------- Initializing the robot chassis constants -----------------##
wheel_base = 0.235
track = 0.15
r = 0.0475
z = 0.0963
body_twist = np.zeros((6,1))
joint_angles = np.zeros((5,1))
wheel_angles = np.zeros((4,1))
F = (r/4) * np.matrix([[-1/(wheel_base + track), 1/(wheel_base + track), 1/(wheel_base + track), -1/(wheel_base + track)],[1,1,1,1],[-1,1,-1,1]])

##------------------------ Function Defination ----------------------##
def Next_State(state, joint_wheel_velocties, tau, max_vel):
    body_twist = np.zeros((6,1))

##-------------------- Sorting Data --------------------##
    theta = state[0]
    x,y = state[1],state[2]
    joint_angles = np.reshape(state[3:8],(5,1))
    wheel_angles = state[8:12].reshape(4,1)
    joint_velocities = joint_wheel_velocties[0:5].reshape(5,1)
    wheel_velocities = joint_wheel_velocties[5:].reshape(4,1)

##-------------------- Limit joint velocities --------------------##
    for i in range(joint_velocities.shape[0]):
        if(abs(joint_velocities[i])>max_vel):
            joint_velocities[i] = max_vel*(abs(joint_velocities[i])/joint_velocities[i])

    for i in range(wheel_velocities.shape[0]):
        if(abs(wheel_velocities[i])>max_vel):
            wheel_velocities[i] = max_vel*(abs(wheel_velocities[i])/wheel_velocities[i])

##-------------------- Find new angles --------------------##
    joint_angles += joint_velocities*tau
    wheel_angles += wheel_velocities*tau

    
##----------------------- Current Pose --------------------##
    current_pose = np.matrix([[np.cos(theta),-np.sin(theta),0,x],[np.sin(theta),np.cos(theta),0,y],[0,0,1,z],[0,0,0,1]])

##----------------------- Calculate Body Twist --------------------##
    del_theta = np.reshape((wheel_velocities*tau),(4,1))
    body_twist[2:5,:] = F@del_theta

##----------------------- New Pose --------------------##
    new_pose = current_pose@expm(mr.VecTose3(body_twist))

##----------------------- Extract State variables --------------------##
    x = new_pose[0,3]
    y = new_pose[1,3]
    theta = np.arccos(new_pose[0,0])

    
##----------------------- Update State --------------------##
    state[0] = theta
    state[1],state[2] = x,y
    state[3:8] = joint_angles[:,0]
    state[8:12] = wheel_angles[:,0]
    
    return state

##------------------- To Test Next_State - uncomment this part ------------------##
# if __name__ == '__main__':
#     tau = 0.01	
#     total_time = 3 				
#     step = int(total_time//tau) 
#     configuration = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
#     speeds = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 1, 2, 2, 1])    # FR RR RL FL
#     max_speed = 10

#     new_configuration = np.zeros((step, 13))
#     new_configuration[0] = configuration

#     for i in range(0, step):
#         configuration = Next_State(configuration, speeds, tau, max_speed)
#         new_configuration[i][0:12] = configuration[0:12]