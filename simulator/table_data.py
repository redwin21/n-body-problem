import numpy as np
import pandas as pd
from pymongo import MongoClient


def tabularize(collection, steps, size=100, n_bodies=3, com=False):
    
    col = ['sim_id']
    for n in range(1,n_bodies+1):
        col.append(f'm_{n}')
    
    initial = ['rx_','ry_','rz_','vx_','vy_','vz_']
    for step in ([0]+steps):
        for i in initial:
            col.append(i + str(step))
    
    counter = 0
    df = pd.DataFrame(columns=cols)
    for idx, entry in enumerate(collection.find()):
    
        if com:
            sim = entry['sol_com']
        else:
            sim = entry['sol']
        
        for i in range(size):
            
            df_entry = [entry['id']]
            
            rand_idx = np.random.choice(len(sim) - steps[-1] - 1)
            
            for step in ([0]+steps):
                df_entry += sim[step + rand_idx]
            
            df[counter] = df_entry
            counter += 1
    
    return df


def main():
    
    client = MongoClient('localhost', 27017)
    db = client['n-body']
    
    collections = ['samples_2_bodies_3_dim_1_m', 
                  'samples_2_bodies_3_dim_0_m',
                  'samples_3_bodies_3_dim_1_m',
                  'samples_3_bodies_3_dim_0_m']
    
    steps = [10, 100, 1000]
            
    for collection in collections:
        
        df = tabularize(db[collection], steps)
        path = f'../data/{collection}.csv'
        df.to_csv(path)
        
    pass


if __name__ = "__main__":
    main()
        