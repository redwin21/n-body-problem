import numpy as np
import pandas as pd


def get_X_y(df, model_step, use_mass=False, transform=True):
    """
    Convert to a format that can be used for modeling.
    
    Inputs:
        df - the data frame of the data
        model_step - time horizon of target data choice. e.g. 10, 100, or 1000
        use_mass - whether or not to include mass in the output. if not, masses of 1 are assumed
        transform - whether to transform the data. see transform_X_y function
    
    Returns:
        X, and y, the predictor and target data, respectively
    """
    # select the columns to use for X from the dataframe
    X_cols = []
    for col in df.columns:
        if use_mass:
            if "_0" in col or "m" == col[0]:
                X_cols.append(col)
        else:
            if "_0" in col:
                X_cols.append(col)
    
    # select the columns to use for y from the data frame (these end in the time step number)
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
    """
    Transform the data to be from the perspective of the first body,
    which removes 6 features of the data by setting the position and velocity of the first body
    to zero and then removing the first body's position and velocity.
    The first body position and velocity is subtracted from the others.
    This improves model performance with less features.
    
    This function only works for 2 or 3 body data.
    
    Inputs:
        X - data to transform
        n_bodies - how many bodies are in the data
        use_mass - whether to include the mass in the transformation
        
    Returns:
        the transformed data
    """
    
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
    """
    This function performs the inverse transformation of the data. It recenters all data around
    the center of mass, and adds back in the position and velocity of the first body.
    
    Inputs:
        X - the data to inverse transform
        n_bodies - how many bodies are in the data
        use_mass - whether to include the mass in the transformation
        
    Returns:
        the inverse transformed data with the first body information recovered.
    """
    
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
    """
    Part of the inverse transformation, this function calculates the center of mass
    of the bodies and subtracts that from all of them, including the first (which was
    at the center originally), and transforms the position and velocity to be relative
    to that center of mass.
    
    Inputs:
        X - the data to inverse transform
        n_bodies - how many bodies are in the data
        use_mass - whether to include the mass in the transformation
        
    Returns:
        the inverse transformed data with the first body information recovered.
    """
    
    # initialize new size of output data
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
    
    # find center of mass of position and velocity
    for t in range(r_com.shape[0]):
        r_com[t,:] = ((r[t,:].reshape((n_bodies, 3)).T @ m[t,:]) / np.sum(m[t,:]))
        v_com[t,:] = ((v[t,:].reshape((n_bodies, 3)).T @ m[t,:]) / np.sum(m[t,:]))
    
    # transform data back to being relative to center of mass
    for i in range(n_bodies):
        r[:,i*3:i*3+3] = r[:,i*3:i*3+3] - r_com
        v[:,i*3:i*3+3] = v[:,i*3:i*3+3] - v_com
    
    if use_mass:
        out = np.concatenate((m, r, v), axis=1)
    else:
        out = np.concatenate((r, v), axis=1)
        
    return out
        