from simulator import NBodySimulator
from pymongo import MongoClient
import numpy as np


def main():
    """
    Run simulation to generate various datasets.
    """
    
    client = MongoClient('localhost', 27017)
    db = client['n-body']
    n_samples = 10000
    
    n_bodies = [2, 2, 2, 2, 3, 3, 3, 3]
    dim = [2, 2, 3, 3, 2, 2, 3, 3]
    same_m = [True, False, True, False, True, False, True, False]
    
    for i in range(len(n_bodies)):
        name = f'samples_{n_bodies[i]}_bodies_{dim[i]}_dim_{int(same_m[i])}_m'
        collection = db[name]
        generate_data(n_samples, n_bodies[i], dim[i], same_m[i], collection)
    
    
    
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

        t = np.linspace(0, 50, 5000)
        
        if collection is None:
            name = ''
        else:
            name = collection.name
        
        try:
            sim.fit(r, v, m).simulate(t)
            
            
            if sim.ode_info['message'] != 'Integration successful.':
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
                                       'message': sim.ode_info['message']
                                      })
            print(f'Generated simulation {n+1:7.0f} / {n_samples:7.0f} for sample {name}')
            n += 1
        except Exception:
            print('Warning for odeint raised. Simulation skipped.')
            continue
    
    return samples


if __name__ == '__main__':
    main()