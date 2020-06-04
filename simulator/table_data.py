import numpy as np
import pandas as pd
from pymongo import MongoClient
import multiprocessing as mp


def tabularize(collection_name, steps, size=100, n_bodies=3, com=False, path=None):
    """
    Turn the mongodb data into a usable dataframe for modeling, randomly sampling from each simulation.
    
    Inputs:
        collection_name - string of name of mongodb collection
        steps - list of steps to look at following current point for modeling
                e.g. for a simulation, if value 0 is sampled, and steps is 100,
                then value at step 100 is added to that dataframe row
        size - number of samples from each simulation
        n_bodies - number of bodies of the simulation
        com - whether to sample center of mass data or not
        path - filepath to save dataframe
        
    Returns:
        dataframe of data with columns corresponding to initial data points and stepped points
    """
    # establsih database connection
    client = MongoClient('localhost', 27017)
    db = client['n-body']
    collection = db[collection_name]
    
    # create column titles for data frame
    # [sim_id, m_1, m_2, rx_1_0, ry_1_0, ... vx_2_100,...] 
    col = ['sim_id']
    for n in range(1,n_bodies+1):
        col.append(f'm_{n}')
    
    r_initial = ['rx_','ry_','rz_']
    v_initial = ['vx_','vy_','vz_']
    for step in ([0]+steps):
        for n in range(1,n_bodies+1):
            for i in r_initial:
                col.append(i + str(n) + '_' + str(step))
        for n in range(1,n_bodies+1):
            for i in v_initial:
                col.append(i + str(n) + '_' + str(step))
    
    # initialize
    counter = 0
    df = pd.DataFrame(columns=col)
    name = collection.name
    
    # loop through each simulation in database
    for idx, entry in enumerate(collection.find()):
    
        # select center of mass data or not
        if com:
            sim = entry['sol_com']
        else:
            sim = entry['sol']
        
        # randomly sample from the simulation size number of times
        for i in range(size):
            
            # get id and mass
            df_entry = [entry['id']]
            for n in range(n_bodies):
                df_entry.append(entry['mass'][n][0])
            
            # get random sample for starting point and each steps point after
            rand_idx = np.random.choice(len(sim) - steps[-1] - 1)
            for step in ([0]+steps):
                df_entry += sim[step + rand_idx]
            
            print(f'Data point {counter:6.0f} of table {name}')
            
            # add combined row to dataframe
            df.loc[counter] = df_entry
            counter += 1
      
    if not path is None:
        df.to_csv(path, index=False)
    
    return df




def main():
    """
    Run to generate csv's of data from each type of simulation.
    Parallelized.
    """
    
    collections = ['samples_2_bodies_3_dim_1_m', 
                  'samples_2_bodies_3_dim_0_m',
                  'samples_3_bodies_3_dim_1_m',
                  'samples_3_bodies_3_dim_0_m']
    
    steps = [10, 100, 1000]
    processes = []
            
    for collection in collections:
        
        n_bodies = int(collection.split('_')[1])
        path = f'./data/{collection}_com.csv'
        
        p = mp.Process(target=tabularize, args=(collection, steps, 100, n_bodies, False, path))
        processes.append(p)
        p.start()
    
    for process in processes:
        process.join()
        
    pass


if __name__ == "__main__":
    main()
        