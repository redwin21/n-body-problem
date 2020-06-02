import numpy as np
from scipy.integrate import odeint
import warnings
from pymongo import MongoClient


class NBodySimulator:
    """
    Class to generate a simulation of the n-body problem using odeint solver to solve the equations with n bodies.
    """
    
    def __init__(self, n, G=1):
        """
        Initialize the n-body simulator with n bodies.
        
        Inputs:
            n - number of bodies in the system
            G - gravitational constant. default is 1.
        """
        
        self.n = n
        self.G = G
        self.m = None
        self.r = None
        self.v = None
        self.r_com = None
        self.v_com = None
        self.sol = None
        self.ode_info = None
        self.r_sol = None
        self.v_sol = None
        self.r_sol_com = None
        self.v_sol_com = None
        
    
    def fit(self, r, v, m=None):
        """
        Fit the simulator with the masses of each body, as well as the initial positions of 
        r and v.
        
        Inputs:
            m - np.array of masses of each body. 
                Should be shape (n*3,1).
            r - np.array of initial positions of each body. 
                Should be shape (n*3,3). 
                i.e. [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]] for 3 bodies.
                Set z's to 0 for 2D simulations.
            v - np.array of initial velocities of each body. 
                Should be shape (n*3,3). 
                i.e. [[vx1, vy1, vz1], [vx2, vy2, vz2], [vx3, vy3, vz3]] for 3 bodies.
                Set z's to 0 for 2D simulations.
        """
        
        req_shape_m = (self.n,1)
        req_shape_r = (self.n,3)
        req_shape_v = (self.n,3)
        
        if m is None:
            m = np.full(shape=req_shape_m, fill_value=1)
        
        m = m.reshape(req_shape_m)
        
        if not m.shape == req_shape_m:
            raise ValueError(f'Incorrect size of m for {self.n} bodies. Should be array of shape {req_shape_m}.')
        if not r.shape == req_shape_r:
            raise ValueError(f'Incorrect size of r for {self.n} bodies. Should be array of shape {req_shape_r}.')
        if not v.shape == req_shape_v:
            raise ValueError(f'Incorrect size of v for {self.n} bodies. Should be array of shape {req_shape_v}.')
            
        self.m = m
        self.r = r
        self.v = v
        
        return self
        
    
    def simulate(self, t):
        """
        Use ODE solver to generate solution.
        
        Inputs:
            t - np.array of time steps to solve for.
                First index should correspond to initial time.
        """
        
        init_params = np.concatenate((self.r.flatten(), self.v.flatten()))
        
        sol = odeint(solution, init_params, t, args=(self.n, self.G, self.m), full_output=1)
        self.sol = sol[0]
        self.ode_info = sol[1]
        self.r_sol = self.sol[:,:self.n*3]
        self.v_sol = self.sol[:,self.n*3:]
        
        self.normalize_com()
    
    
    def normalize_com(self):
        """
        Creates a new set of solution results that is 'normalized' by subtracting off the 
        center of mass so the system can be plotted and viewd from a stationary point.
        """
        
        self.r_com = np.zeros(shape=(self.r_sol.shape[0],3))
        self.v_com = np.zeros(shape=(self.r_sol.shape[0],3))
        for t in range(self.r_com.shape[0]):
            self.r_com[t] = ((self.r_sol[t].reshape(self.r.shape).T @ self.m) / np.sum(self.m)).T
            self.v_com[t] = ((self.v_sol[t].reshape(self.v.shape).T @ self.m) / np.sum(self.m)).T
        
        
        self.r_sol_com = np.zeros(self.r_sol.shape)
        self.v_sol_com = np.zeros(self.v_sol.shape)
        for i in range(self.n):
            self.r_sol_com[:,i*3:i*3+3] = self.r_sol[:,i*3:i*3+3] - self.r_com
            self.v_sol_com[:,i*3:i*3+3] = self.v_sol[:,i*3:i*3+3] - self.v_com
    
    
    pass
   

    
    
def solution(w, t, n, G, m):
    """
    Calculates the derivatives of the n-body solution for numerical integration.
    
    Inputs:
        w - the initial conditions vector (r and v flattened and concatenated)
        t - the number of time steps
        n - the number of bodies
        G - the gravitational constant
        m - np.array of the masses of each body. Should be shape (n*3,1).
        
    Returns:
        np.array of derivatives dr/dt and dv/dt
    """
    
    r = w[:n*3].reshape((n,3))
    
    dvdt = np.zeros(shape=(n,3))
        
    for body in range(n):
        for other_body in range(n):
            if not body == other_body:
                dvdt[body] += G * m[other_body] * (r[other_body] - r[body]) / np.linalg.norm(r[other_body] - r[body]) ** 3
        
    drdt = w[n*3:]
        
    return np.concatenate((drdt, dvdt.flatten()))


def generate_data(n_samples, n_bodies=3, dim=3, same_m=True, collection=None):
    """
    Function to create sample simulations using random initial values.
    
    Inputs:
        n_samples - number of samples
        n_bodies - number of bodies. default 3
        dim - number of dimensions to generate data (can be 2 or 3, raises error if not). default 3
        same_m - boolean of whether masses are the same or not. default True, sets all to 1
        collection - mongodb collection to write data to. default None
        
    Returns:
        list of samples of length n_samples
    """
    
    if not dim == 3 and not dim == 2:
        raise ValueError(f'Incorrect value {dim} for dim. Should be 2 or 3.')
    
    sim = NBodySimulator(n=n_bodies)
    
    samples = []
    n = 0
    while n < n_samples:
        if same_m:
            m = np.ones((n_bodies,1))
        else:
            m = np.random.random((n_bodies,1))*5

        r = np.random.random((n_bodies, 3))*2-1
        v = np.random.random((n_bodies, 3))*2-1

        if dim == 2:
            r[:,2] = 0
            v[:,0] = 0

        t = np.linspace(0, 200, 5000)
        
        if collection is None:
            name = ''
        else:
            name = collection.name
        
        try:
            sim.fit(r, v, m).simulate(t)
            
            if sim.sol_info['message'] != 'Integration successful.':
                raise Exception()
                
            if collection is None:
                samples.append(sim.sol)
            else:
                collection.insert_one({'id': n, 
                                       'sol': sim.sol.tolist(),
                                       'r_sol': sim.r_sol.tolist(),
                                       'v_sol': sim.v_sol.tolist(),
                                       'r_sol_com': sim.r_sol_com.tolist(),
                                       'v_sol_com': sim.v_sol_com.tolist(),
                                       'info': sim.sol_info
                                      })
            print(f'Generated simulation {n+1:7.0f} / {n_samples:7.0f} for sample {name}')
            n += 1
        except Exception:
            print('Warning for odeint raised. Simulation skipped.')
            continue
    
    return samples