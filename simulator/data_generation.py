from simulator import NBodySimulator
from pymongo import MongoClient
import numpy as np
import multiprocessing as mp


def main():
    """
    Run simulation to generate various datasets.
    """
    
    processes = []
    
    client = MongoClient('localhost', 27017)
    db = client['n-body']
    n_samples = 2000
    
    n_bodies = [2, 2, 3, 3]
    dim = [3, 3, 3, 3]
    same_m = [True, False, True, False]
    
    for i in range(len(n_bodies)):
        p = mp.Process(target=generate_data, args=(n_samples,
                                         n_bodies[i], 
                                         dim[i], 
                                         same_m[i], 
                                         f'samples_{n_bodies[i]}_bodies_{dim[i]}_dim_{int(same_m[i])}_m'))
        processes.append(p)
        p.start()
    
    for process in processes:
        process.join()
    
def generate_data(n_samples, n_bodies=3, dim=3, same_m=True, collection_name=None):
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
    
    client = MongoClient('localhost', 27017)
    db = client['n-body']
    
    if collection_name is not None:
        collection = db[collection_name]
    
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

        t = np.linspace(0, 30, 5000)
        
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
                                       'mass': sim.m.tolist(),
                                       'sol': sim.sol.tolist(),
                                       'sol_com': sim.sol_com.tolist(),
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
