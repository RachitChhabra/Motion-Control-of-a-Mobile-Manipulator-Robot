from concurrent.futures import thread
from termios import TAB1
import numpy as np
import modern_robotics as mr

##------------- Initializing the robot chassis constants -----------------##
Tb0 = np.array([[1,0,0,0.1662],[0,1,0,0],[0,0,1,0.0026],[0,0,0,1]])
M0e = np.array([[1,0,0,0.0330],[0,1,0,0],[0,0,1,0.6546],[0,0,0,1]])
wheel_base = 0.235
track = 0.15
r = 0.0475
z = 0.0963
F = (r/4) * np.matrix([[-1/(wheel_base + track), 1/(wheel_base + track), 1/(wheel_base + track), -1/(wheel_base + track)],[1,1,1,1],[-1,1,-1,1]])
F6 = np.vstack((np.zeros((2,4)),F,np.zeros((1,4))))

##----------------------- Function Defination --------------------##
def Feedback_Control(X, Xd, Xd_next, Kp, Ki, tau, B_list, theta_list,X_error_int):

##------------- Forward Kinematics -----------------##
    T0e = mr.FKinBody(M0e, B_list, theta_list)
    Tbe = np.matmul(Tb0,T0e)

##------------- Calculation Body Jacobian -----------------##
    Teb = mr.TransInv(Tbe)

    J_base = np.matmul(mr.Adjoint(Teb),F6)
    J_arm  = mr.JacobianBody(B_list,theta_list)
    Je     = np.hstack((J_base,J_arm))     ## DIFF
    Je_inv = np.linalg.pinv(Je,1e-3)

##------------- Calculation Error Twist -----------------##
    X_error_f = mr.se3ToVec(mr.MatrixLog6(np.array(np.matmul(mr.TransInv(X),Xd))))
    X_error_f = np.reshape(X_error_f,(6,1))

    X_error_int = np.reshape(X_error_int,(6,1))
    X_error_int += X_error_f*tau

##------------------- Desired Twist ------------------##
    Vd = mr.se3ToVec(mr.MatrixLog6(np.array(np.linalg.inv(Xd)@Xd_next)))/tau

    Adj = mr.Adjoint(np.matmul(np.linalg.inv(X),Xd))

##------------------- Calculating Required Twist ------------------##
    feedforward = np.matmul(Adj,Vd)
    feedforward = np.reshape(feedforward,(6,1))

    V = feedforward + np.matmul(Kp,X_error_f) + np.matmul(Ki,X_error_int)

##------------------- Calculating Joints and Wheels Velocities ------------------##
    theta_dot = np.matmul(Je_inv,V)

    return Vd, V, Je, theta_dot, X_error_f, X_error_int


##------------------- To Test Feedback_Control- uncomment this part ------------------##
# if __name__ == '__main__':
#     Xd      = np.array([[0,0,1,0.5],[0,1,0,0],[-1,0,0,0.5],[0,0,0,1]])
#     Xd_next = np.array([[0,0,1,0.6],[0,1,0,0],[-1,0,0,0.3],[0,0,0,1]])
#     X       = np.array([[0.17,0,0.985,0.387],[0,1,0,0],[-0.985,0,0.17,0.57],[0,0,0,1]])
    
#     Kp = np.eye(6)
#     Ki = np.zeros((6,6))
#     B_list = np.array([[0,0,1,0,0.0330,0],[0,-1,0,-0.5076,0,0],[0,-1,0,-0.3526,0,0],[0,-1,0,-0.2176,0,0],[0,0,1,0,0,0]]).transpose()
#     theta_list = np.array([[0,0,0.2,-1.6,0]]).transpose()
#     X_error_int = np.zeros((6,1))

#     tau = 0.01

#     Feedback_Control(X, Xd, Xd_next, Kp, Ki, tau, B_list, theta_list,X_error_int)

    

