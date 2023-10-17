import numpy as np
import numba as nb
import matplotlib.pyplot as plt


# 基础参数
v = 0.03
# 速度
alpha = 1
# 耦合强度上的系数




def KM(omega,theta,c1,c2):
    d_theta = omega + 0.1*(np.cos(theta) * c1 - np.sin(theta) * c2)
    return d_theta

@nb.jit(nopython=True)
def Vicsek(theta,x,y,v):
    x = x + v * np.cos(theta)
    y = y + v * np.sin(theta)
    x = np.mod(x,10)
    y = np.mod(y,10)
    return x,y

@nb.jit(nopython=True)
def matrix(x,y):
    N = len(x)
    R = np.zeros((N,N))
    for i in range(len(x)):
        for j in range(len(x)):
            R[i,j]=((x[i]-x[j])**2+(y[i]-y[j])**2)**0.5
    x = np.mod(x,10)
    y = np.mod(y,10)
    return R,x,y

# exp(-alpha*r)
@nb.jit(nopython=True)
def K_exp(R,theta):
    K = np.exp(-alpha*R)
    c1 = np.dot(K, np.sin(theta))
    c2 = np.dot(K, np.cos(theta))
    return c1,c2

# 1/(exp(-alpha*r)+1)
@nb.jit(nopython=True)
def K_1exp(R,theta):
    K = 1/(1+np.exp(alpha*R))
    c1 = np.dot(K, np.sin(theta))
    c2 = np.dot(K, np.cos(theta))
    return c1,c2


@nb.jit(nopython=True)
def K_matrix(x,y,theta):
    c1 = []
    c2 = []
    for j in range(len(x)):
        C1 = 0.0
        C2 = 0.0
        for k in range(len(x)):
            D = ((x[j] - x[k]) ** 2 + (y[j] - y[k]) ** 2) ** 0.5
            if D > 1:
                a = 0
            else:
                a = 1
            C1 = C1 + a * np.sin(theta[k])
            C2 = C2 + a * np.cos(theta[k])
        c1.append(C1)
        c2.append(C2)
    return c1,c2

def runge_kutta(w,theta,t,x,y):
    N = len(theta)
    R = []
    dt =0.01
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    Theta = []
    Theta1_c = []
    Theta1_s = []
    Theta2_c = []
    Theta2_s = []
    for i in range(t):
        mmmm1 = i
        x,y = Vicsek(theta,x,y,v)
        c1,c2 = K_matrix(x,y,theta)
        print(i)
        # K = strength(x,y)
        k1 = KM(w, theta, c1, c2)
        k2 = KM(w, theta + k1 * 0.5 * dt,c1,c2)
        k3 = KM(w, theta + k2 * 0.5 * dt,c1,c2)
        k4 = KM(w, theta + k3 * dt,c1,c2)
        theta = theta + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        for i in range(len(theta)):
            if theta[i]>np.pi:
                theta[i]=theta[i]-2*np.pi
            elif theta[i]<-np.pi:
                theta[i]=theta[i]+2*np.pi
        L1 =[]
        L2 =[]
        for i in range(len(k1)):
            if k1[i]>=0:
                K1 = list(k1)
                L1.append(K1.index(k1[i]))
            else:
                K1 = list(k1)
                L2.append(K1.index(k1[i]))
        r =((np.sum(np.sin(theta))/N)**2+(np.sum(np.cos(theta))/N)**2)**0.5
        R.append(r)
        Theta.append(theta)
        Theta1_c.append(np.cos(theta[L1]))
        Theta2_c.append(np.cos(theta[L2]))
        Theta1_s.append(np.sin(theta[L1]))
        Theta2_s.append(np.sin(theta[L2]))
        X1.append(x[L1])
        Y1.append(y[L1])
        X2.append(x[L2])
        Y2.append(y[L2])

    return R, x, y, theta,X1,X2,Y1,Y2,Theta,Theta1_c,Theta1_s,Theta2_c,Theta2_s

def runge_kutta_K_exp(w,theta,t,x,y):
    N = len(theta)
    R = []
    dt =0.01
    for i in range(t):
        x,y = Vicsek(theta,x,y,v)
        K,x,y = matrix(x,y)
        c1,c2 = K_exp(K,theta)
        print(i)
        # K = strength(x,y)
        k1 = KM(w, theta, c1, c2)
        k2 = KM(w, theta + k1 * 0.5 * dt,c1,c2)
        k3 = KM(w, theta + k2 * 0.5 * dt,c1,c2)
        k4 = KM(w, theta + k3 * dt,c1,c2)
        theta = theta + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        for i in range(len(theta)):
            if theta[i]>np.pi:
                theta[i]=theta[i]-2*np.pi
            elif theta[i]<-np.pi:
                theta[i]=theta[i]+2*np.pi
        r =((np.sum(np.sin(theta))/N)**2+(np.sum(np.cos(theta))/N)**2)**0.5
        R.append(r)
    return R,x,y,theta


def runge_kutta_K1_exp(w,theta,t,x,y):
    N = len(theta)
    R = []
    dt =0.01
    for i in range(t):
        x,y = Vicsek(theta,x,y,v)
        K,x,y = matrix(x,y)
        c1,c2 = K_1exp(K,theta)
        print(i)
        # K = strength(x,y)
        k1 = KM(w, theta, c1, c2)
        k2 = KM(w, theta + k1 * 0.5 * dt,c1,c2)
        k3 = KM(w, theta + k2 * 0.5 * dt,c1,c2)
        k4 = KM(w, theta + k3 * dt,c1,c2)
        theta = theta + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        for i in range(len(theta)):
            if theta[i]>np.pi:
                theta[i]=theta[i]-2*np.pi
            elif theta[i]<-np.pi:
                theta[i]=theta[i]+2*np.pi
        r =((np.sum(np.sin(theta))/N)**2+(np.sum(np.cos(theta))/N)**2)**0.5
        R.append(r)
    return R,x,y,theta