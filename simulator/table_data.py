import numpy as np
import pandas as pd
from pymongo import MongoClient
import multiprocessing as mp


def tabularize(collection_name, steps, size=100, n_bodies=3, com=False, path=None):
    
    client = MongoClient('localhost', 27017)
    db = client['n-body']
    collection = db[collection_name]
    
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
    
    counter = 0
    df = pd.DataFrame(columns=col)
    name = collection.name
    
    for idx, entry in enumerate(collection.find()):
    
        if com:
            sim = entry['sol_com']
        else:
            sim = entry['sol']
        
        for i in range(size):
            
            df_entry = [entry['id']]
            for n in range(n_bodies):
                df_entry.append(entry['mass'][n][0])
            
            rand_idx = np.random.choice(len(sim) - steps[-1] - 1)
            
            for step in ([0]+steps):
                df_entry += sim[step + rand_idx]
            
            print(f'Data point {counter:6.0f} of table {name}')
            
            df.loc[counter] = df_entry
            counter += 1
            
    if not path is None:
        df.to_csv(path, index=False)
    
    return df




def main():
    
    collections = ['samples_2_bodies_3_dim_1_m', 
                  'samples_2_bodies_3_dim_0_m',
                  'samples_3_bodies_3_dim_1_m',
                  'samples_3_bodies_3_dim_0_m']
    
    steps = [10, 100, 1000]
    processes = []
            
    for collection in collections:
        
        n_bodies = int(collection.split('_')[1])
        path = f'./data/{collection}.csv'
        
        p = mp.Process(target=tabularize, args=(collection, steps, 100, n_bodies, False, path))
        processes.append(p)
        p.start()
    
    for process in processes:
        process.join()
        
    pass


if __name__ == "__main__":
    main()
        