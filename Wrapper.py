import numpy as np
import modern_robotics as mr
from Feedback_Control import Feedback_Control
from TrajectoryGenerator import get_trajectory
from Next_State import Next_State
import matplotlib.pyplot as plt

##------------- Initializing the robot chassis constants -----------------##
wheel_base = 0.235
track = 0.15
r = 0.0475
z = 0.0963
configuration = np.array([-np.pi/3, 0.1, -0.2, 0, 0, -0.7, -1.6 , 0, 0, 0, 0, 0, 0])
Tb0 = np.array([[1,0,0,0.1662],[0,1,0,0],[0,0,1,0.0026],[0,0,0,1]])
M0e = np.array([[1,0,0,0.0330],[0,1,0,0],[0,0,1,0.6546],[0,0,0,1]])
B_list = np.array([[0,0,1,0,0.0330,0],[0,-1,0,-0.5076,0,0],[0,-1,0,-0.3526,0,0],[0,-1,0,-0.2176,0,0],[0,0,1,0,0,0]]).transpose()

Tse = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0.5],[0,0,0,1]])

##----------------- Initialising Feedback Gains -----------------##
Kp = np.eye(6)*3
Ki = np.eye(6)*7

tau = 0.01

max_velocity = 10

##------------- Getting Trajectory from TrajectoryGenerator -----------------##
trajectory = get_trajectory()

Total_Error = np.zeros((trajectory.shape[0],6))
config_array = np.zeros((trajectory.shape[0],13))
X_error  = np.zeros((6,1))
X_error_int = np.zeros((6))

##--------------- Assigning starting configuration -----------------##
config_array[0] = configuration


for i in range(1,trajectory.shape[0]-1):
    
##---------------- Actual/Current Configuration  ------------------##
    theta = config_array[i-1,0]
    x,y = config_array[i-1,1],config_array[i-1,2]
    Tsb = np.array([[np.cos(theta),-np.sin(theta),0,x],[np.sin(theta),np.cos(theta),0,y],[0,0,1,z],[0,0,0,1]])

    T0e = mr.FKinBody(M0e ,B_list, config_array[i-1,3:8])
    
    Tbe = np.matmul(Tb0,T0e)
    X_orig = np.matmul(Tsb,Tbe)

##-------------------- Desired Configuration -------------------##
    Xd_temp = np.zeros((3,3))
    Xd_temp[0,0:3] = trajectory[i,0:3]
    Xd_temp[1,0:3] = trajectory[i,3:6]
    Xd_temp[2,0:3] = trajectory[i,6:9]
    Xd = np.vstack([np.hstack((Xd_temp,np.transpose(np.reshape(trajectory[i,9:12],(1,3))))),np.matrix([0,0,0,1])])

    Xd_temp = np.zeros((3,3))
    Xd_temp[0,0:3] = trajectory[i+1,0:3]
    Xd_temp[1,0:3] = trajectory[i+1,3:6]
    Xd_temp[2,0:3] = trajectory[i+1,6:9]
    Xd_next = np.vstack([np.hstack((Xd_temp,np.transpose(np.reshape(trajectory[i+1,9:12],(1,3))))),np.matrix([0,0,0,1])])


    theta_list = config_array[i-1,3:8]

##-------------------- Feedback Control -------------------##
    Vd, V, Je, theta_dot, X_error, X_error_int = Feedback_Control(X_orig, Xd, Xd_next, Kp, Ki, tau, B_list, theta_list,X_error_int)
 
##-------------------- Stacking Velocities -------------------##
    theta_dot = np.vstack((theta_dot[4:],theta_dot[0:4]))

##-------------------- Next_State --------------------------##
    config_array[i] = Next_State(config_array[i-1,:], theta_dot, tau, max_velocity)

##--------------------- Total Error ------------------------##
    Total_Error[i] = X_error.reshape(6,)

##------------------- Assigning End Effector Gripper State ---------------------##
config_array[:,-1] = trajectory[:,-1]

##------------------------ Plotting Error ---------------------##
ac = np.linspace(0,Total_Error.shape[0]-1,Total_Error.shape[0])

plt.xlabel("Iterations")
plt.ylabel("Error")
plt.plot(ac,Total_Error[:,0])
plt.plot(ac,Total_Error[:,1])
plt.plot(ac,Total_Error[:,2])
plt.plot(ac,Total_Error[:,3])
plt.plot(ac,Total_Error[:,4])
plt.plot(ac,Total_Error[:,5])
plt.legend(["wx","wy","wz","vx","vy","vz"])
plt.show(block = True)

#------------- Save to CSV -----------------##
np.savetxt('final.csv',config_array,delimiter = ",")

