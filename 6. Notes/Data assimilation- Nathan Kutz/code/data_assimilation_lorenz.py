#import os  # clear screen
import numpy as np   # matrix calc
from scipy.integrate import odeint # scientific computation (ode solver)

import matplotlib.pyplot as plt
#os.system('clear')

def lorenz(y,t,sigma,beta,rho):
    dy=[sigma*(y[1]-y[0]),y[0]*(rho-y[2])-y[1],y[0]*y[1]-beta*y[2]]
    return dy
    


# Lorenz parameters (chaotic)
sigma=10
beta=8/3
rho=28
tspan=np.arange(0,20.01,0.01,dtype=float)  # time span
x0= np.array([5,5,5])                 # initial conditions


x=odeint(lambda x,t:lorenz(x,t,sigma,beta,rho), x0, tspan, atol=1e-10, rtol=1e-10)
print('ODEint successfully ran for exact solution')

# plotting
fig = plt.figure(1)

ax = fig.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],"black")   # disp-X vs disp-y for exact solution

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.view_init(9,142)   

plt.figure(2)
plt.plot(tspan,x[:,0],'black')

# Adding noise
q2=1 # strength of noise
fig=plt.figure(3)
for i in range(1,9):
    # Purturbed initial condition
    x_ic=x0+q2*np.random.randn(3)  # randn(mu,sigma^2) adding white noise
    x_noisy=odeint(lambda x,t:lorenz(x,t,sigma,beta,rho), x_ic, tspan, atol=1e-10, rtol=1e-10)
    plt.subplot(4,2,i)   
    plt.plot(tspan,x[:,0],'black',tspan,x_noisy[:,0],'red')
fig.suptitle('Effect of noisy initial data') 
    
# Noisy observations for observation data
# Selecting every 50 timestep for clear plot
tdata=tspan[::50]
n=np.shape(tdata)[0]
# White noise: normal distribution
xn=np.random.randn(n)
yn=np.random.randn(n)
zn=np.random.randn(n)

q3=1.5 # strength of the error

xdata=x[::50,0]+q3*xn
ydata=x[::50,1]+q3*yn
zdata=x[::50,2]+q3*zn
# plotting the time series data
plt.figure(4)
plt.title('Time-series')
plt.plot(tspan,x[:,0],'black',tdata,xdata,'ro') # use 'ro' to plot a scatter

# Data assimilation
x_da=np.array([])
for j in range(0,np.shape(tdata)[0]-1): # data assimilation in steps
    tspan2=np.arange(0,0.51,0.01,dtype=float)
    # taking out model predictions for tspan2
    x_sol=odeint(lambda x,t:lorenz(x,t,sigma,beta,rho), x_ic, tspan2, atol=1e-10, rtol=1e-10)
    # choosing the last point of the exact solution as the initial condition
    # Model predictions
    xic0=np.hstack([x_sol[-1,0],x_sol[-1,1],x_sol[-1,2]])
    # new x pivot
    # Measurements
    xdat=np.hstack([xdata[j+1],ydata[j+1],zdata[j+1]])
    K=q2/(q2+q3) # Kalman filter
    # Updated initial condition
    x_ic=xic0+K*(xdat-xic0)
    # to vertically concatenate the [x,y,z]
    # if x_da.size is false ther concatenation won't work because of inconsistent dimension
    # so we just replace x_da with x_sol if x_da.size is false
    # Here -1 is used to ignore the last point overlapping.
    x_da=np.vstack([x_da,x_sol[:-1,:]]) if x_da.size else x_sol[:-1,:]

# Concatenate the last point for the last iteration
x_da=np.vstack([x_da,x_sol[-1,:]])
plt.figure(5)
plt.title('EKF with purturbed initial condition and noisy data')
plt.plot(tspan,x[:,0],'black',tspan[:-1],x_da[:,0],'red')
    
    

