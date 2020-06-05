import numpy as np
import pandas as pd


def get_X_y(df, model_step, use_mass=False, transform=True):
    X_cols = []
    for col in df.columns:
        if use_mass:
            if "_0" in col or "m" == col[0]:
                X_cols.append(col)
        else:
            if "_0" in col:
                X_cols.append(col)

    check = '_' + str(model_step)
    y_cols = []
    for col in df.columns:
        if check == col[-len(check):]:
            y_cols.append(col)

    X = df[X_cols].values
    y = df[y_cols].values
    n_bodies = y.shape[1]/6
    
    if transform:
        X = transform_X_y(X, n_bodies, use_mass)
        y = transform_X_y(y, n_bodies, use_mass=False)
    
    return X, y
    

def transform_X_y(X, n_bodies, use_mass=False):
    
    new = np.zeros((X.shape[0],X.shape[1]-6))
    
    if n_bodies == 2:
        if use_mass:
            # m
            new[:,0:2] = X[:,0:2]
            # new r
            new[:,2:5] = X[:,5:8] - X[:,2:5]
            #new v
            new[:,5:8] = X[:,11:14] - X[:,8:11]
        else:
            # new r
            new[:,:3] = X[:,3:6] - X[:,:3]
            #new v
            new[:,3:6] = X[:,9:12] - X[:,6:9]
            
    elif n_bodies == 3:
        if use_mass:
            # m
            new[:,0:3] = X[:,0:3]
            # new r1 and r2
            new[:,3:6] = X[:,6:9] - X[:,3:6]
            new[:,6:9] = X[:,9:12] - X[:,3:6]
            #new v1 and v2
            new[:,9:12] = X[:,15:18] - X[:,12:15]
            new[:,12:15] = X[:,18:21] - X[:,12:15]
        else:
            # new r1 and r2
            new[:,:3] = X[:,3:6] - X[:,:3]
            new[:,3:6] = X[:,6:9] - X[:,:3]
            #new v1 and v2
            new[:,6:9] = X[:,12:15] - X[:,9:12]
            new[:,9:12] = X[:,15:18] - X[:,9:12]
            
    else:
        raise ValueError('n_bodies must be 2 or 3')
        
    return new


def inverse_transform_X_y(X, n_bodies, use_mass=False):
    
    new = np.zeros((X.shape[0],X.shape[1]+6))
    
    if n_bodies == 2:
        if use_mass:
            # mass
            new[:,0:2] = X[:,0:2]
            # new r
            new[:,5:8] = X[:,2:5]
            #new v
            new[:,11:14] = X[:,5:8]
        else:
            # new r
            new[:,3:6] = X[:,0:3]
            #new v
            new[:,9:12] = X[:,3:6]
            
    elif n_bodies == 3:
        if use_mass:
            # mass
            new[:,0:3] = X[:,0:3]
            # new r
            new[:,6:12] = X[:,3:9]
            #new v
            new[:,15:21] = X[:,9:15]
        else:
            # new r
            new[:,3:9] = X[:,0:6]
            #new v
            new[:,12:18] = X[:,6:12]
            
    else:
        raise ValueError('n_bodies must be 2 or 3')
        
    return normalize(new, n_bodies, use_mass)

        
def normalize(X, n_bodies, use_mass=False):
    
    if use_mass:
        r_com = np.zeros(shape=(X.shape[0],3))
        v_com = np.zeros(shape=(X.shape[0],3))
        m = X[:,:n_bodies]
        if n_bodies == 2:
            r = X[:,2:8]
            v = X[:,8:14]
        elif n_bodies == 3:
            r = X[:,3:12]
            v = X[:,12:21]
        else:
            raise ValueError('n_bodies must be 2 or 3')
            
    else:
        r_com = np.zeros(shape=(X.shape[0],3))
        v_com = np.zeros(shape=(X.shape[0],3))
        m = np.ones(shape=(X.shape[0],n_bodies))
        if n_bodies == 2:
            r = X[:,0:6]
            v = X[:,6:12]
        elif n_bodies == 3:
            r = X[:,0:9]
            v = X[:,9:18]
        else:
            raise ValueError('n_bodies must be 2 or 3')
    
    for t in range(r_com.shape[0]):
        r_com[t,:] = ((r[t,:].reshape((n_bodies, 3)).T @ m[t,:]) / np.sum(m[t,:]))
        v_com[t,:] = ((v[t,:].reshape((n_bodies, 3)).T @ m[t,:]) / np.sum(m[t,:]))
        
    for i in range(n_bodies):
        r[:,i*3:i*3+3] = r[:,i*3:i*3+3] - r_com
        v[:,i*3:i*3+3] = v[:,i*3:i*3+3] - v_com
    
    if use_mass:
        out = np.concatenate((m, r, v), axis=1)
    else:
        out = np.concatenate((r, v), axis=1)
        
    return out
        