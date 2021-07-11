#Imports
import numpy as np
import math
from scipy.linalg import norm
import matplotlib.pyplot as plt

#Define Useful functions
def lorenz_DA(v):
    return np.array([0, (rho * v[0] - v[1] - v[0] * v[2]),
                      (v[0] * v[1] - beta * v[2])])


def lorenz(state):
    return np.array([sigma * (state[1] - state[0]),
            rho * state[0] - state[1] - state[0] * state[2],
            state[0] * state[1] - beta * state[2]])


def update_cond(v, ti, tn, dt, relax, tol):
    return abs(v[1] - v[0]) > tol and dt * (ti - tn) >= relax

#Divided Differences
def diff(f):
    out = np.zeros(len(f) - 1)
    for i in range(len(f) - 1):
        f = f[1:] - f[:-1]
        out[i] = f[-1]
    return out


def deriv(f, dt):
    n = len(f) 
    difs = diff(f)
    tot = 0
    for i in range(1, n - 1):
        tot +=  (1 / i) * difs[i - 1]
    return tot / dt

#Parameters
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

# Time Stepping
t0 = 0
tf = 30
ti = 0
dt = 0.001
t = np.arange(t0, tf, dt)
n = len(t)

#initial guess
sigma_DA = 108

#Relax
relax = .1


#Tolerance
tol = 0.01

#Initial state vectors
u = np.array([60.0, -60.0, 10.0])
v = np.zeros((n,3))
v[0] = np.array([60.0, 0, 0])

#Error Initialize
e_sol = np.zeros(n)
e_sigma = np.zeros(n)

#Last Parameter Update Time Initialize
tn = t0

#Initialize Ledger
vseries = np.array([])


for ti in range(n):
    time = dt * ti
    
    #Add v1 to ledger
    vseries = np.append(vseries, v[ti,0])
    
    #Calculate Error
    error = norm(v[ti]-u)
    if error == 0:
        print('done')
        break
    e_sol[ti] = error
    e_sigma[ti] = abs(sigma_DA - sigma)
        
    #Update Parameter
    if update_cond(v[ti], ti , tn , dt, relax, tol):
        sigma_DA =  deriv(vseries[-3:], dt) / (v[ti - 1, 1] - v[ti - 1, 0])
        tn = ti
        vseries = np.array([])
    
    #Update Nudged Systsen (Forward Euler)
    v[ti + 1] = v[ti] + dt * lorenz_DA(v[ti])
    
    #Update Actual System
    u += dt * lorenz(u)
    
    #Specify exact value
    v[ti + 1, 0] = u[0]
    
# Plotting the graph
plt.plot(t, e_sigma, color='red', lw=2, label = r'$|\Delta \sigma|$')
plt.plot(t, e_sol, color='blue', lw=2, label = r'$|(u,v,w)|$')

# Setting a logarithmic scale for y-axis
plt.yscale('log')

#Plot Legend
plt.legend()

#Plot Table
plt.title('Exact replacement')

plt.show()