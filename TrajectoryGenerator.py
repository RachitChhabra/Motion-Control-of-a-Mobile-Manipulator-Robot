import numpy as np
import modern_robotics as mr

Tf = 4
k = 1
tau = 0.01
N = int(Tf*k/(tau))
method = 3

trajectory = np.zeros((int(4.4*N),13))


##------------- Initialising se(3) matrices for end effector postion at every step -----------------##
Xstart = np.array([[0 , 0, 1, 0.4],
                   [0 , 1, 0, 0],
                   [-1, 0, 0, 0.4],
                   [0 , 0, 0, 1]])


X1 = np.array([[-0.70710678,  0,          0.70710678,    1  ],
               [ 0,           1,          0,             0  ],
               [-0.70710678,  0,         -0.70710678,  0.2  ],
               [ 0,           0,          0,             1. ]])


X2 = np.array([[-0.70710678,  0,          0.70710678,      1   ],
               [ 0,           1,          0,               0   ],
               [-0.70710678,  0,         -0.70710678,  0.025   ],
               [ 0,           0,          0,               1   ]])

X4   = X1

X5 = np.array([ [ 0,          1,          0,             0 ],
                [0.70710678,  0,         -0.70710678,   -1 ],
                [-0.70710678, 0,         -0.70710678,  0.1 ],
                [ 0,          0,          0,             1 ]])

X6 = np.array([ [ 0,          1,          0,               0 ],
                [0.70710678,  0,          -0.70710678,    -1 ],
                [-0.70710678, 0,         -0.70710678,  0.025 ],
                [ 0,          0,          0,               1 ]])

X8 = X5


##------------- Trajectory 1 -----------------##
T1 = np.array(mr.ScrewTrajectory(Xstart, X1, Tf, N, method))
R1 = T1[:,0:3,0:3].reshape(N,9)
p1 = T1[:,0:3,-1]
Traj1 = np.hstack([R1,p1,np.zeros((N,1))])

trajectory[0:N,:] = Traj1

##------------- Trajectory 2 -----------------##
T2 = np.array(mr.ScrewTrajectory(X1, X2, Tf, int(0.5*N), method))
R2 = T2[:,0:3,0:3].reshape(int(0.5*N),9)
p2 = T2[:,0:3,-1]
Traj2 = np.hstack([R2,p2,np.zeros((int(0.5*N),1))])

trajectory[N:int(1.5*N),:] = Traj2

##------------- Trajectory 3 -----------------##
T3 = Traj2[-1]
Traj3 = T3
Traj3[-1] = 1

trajectory[int(1.5*N):int(1.7*N),:] = Traj3

##------------- Trajectory 4 -----------------##
T4 = np.array(mr.ScrewTrajectory(X2, X4, Tf, int(0.5*N), method))
R4 = T4[:,0:3,0:3].reshape(int(0.5*N),9)
p4 = T4[:,0:3,-1]

Traj4 = np.hstack([R4,p4,np.ones((int(0.5*N),1))])

trajectory[int(1.7*N):int(2.2*N),:] = Traj4

##------------- Trajectory 5 -----------------##
T5 = np.array(mr.ScrewTrajectory(X4, X5, Tf, N, method))
R5 = T5[:,0:3,0:3].reshape(N,9)
p5 = T5[:,0:3,-1]

Traj5 = np.hstack([R5,p5,np.ones((N,1))])

trajectory[int(2.2*N):int(3.2*N),:] = Traj5

##------------- Trajectory 6 -----------------##
T6 = np.array(mr.ScrewTrajectory(X5, X6, Tf, int(0.5*N), method))
R6 = T6[:,0:3,0:3].reshape(int(0.5*N),9)
p6 = T6[:,0:3,-1]

Traj6 = np.hstack([R6,p6,np.ones((int(0.5*N),1))])

trajectory[int(3.2*N):int(3.7*N),:] = Traj6

##------------- Trajectory 7 -----------------##
T7 = Traj6[-1]
Traj7 = T7
Traj7[-1] = 0
trajectory[int(3.7*N):int(3.9*N),:] = Traj7

##------------- Trajectory 8 -----------------##
T8 = np.array(mr.ScrewTrajectory(X6, X8, Tf, int(0.5*N), method))
R8 = T8[:,0:3,0:3].reshape(int(0.5*N),9)
p8 = T8[:,0:3,-1]

Traj8 = np.hstack([R8,p8,np.zeros((int(0.5*N),1))])

trajectory[int(3.9*N):int(4.4*N),:] = Traj8

def get_trajectory():
    return trajectory

#------------- Save to CSV -----------------##
np.savetxt('trajectory.csv', trajectory, delimiter = ',')