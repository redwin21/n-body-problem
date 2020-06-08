import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
from transform import get_X_y, transform_X_y, inverse_transform_X_y, normalize

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


class GSModel():
    
    def __init__(self, n_bodies, steps, parameters, use_mass=False):
        
        self.model = None
        self.parameters = parameters
        self.n_bodies = n_bodies
        self.steps = steps
        self.use_mass = use_mass
        
        
    def fit(self, X, y):

        self.model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=1000,
                                                                   learning_rate=0.05,
                                                                   max_depth=10), n_jobs=-1)
#         self.model = GridSearchCV(estimator=model, 
#                                 param_grid=self.parameters, 
#                                 cv=5, 
#                                 scoring='neg_mean_squared_error',
#                                 n_jobs=-1
#                             )

        self.model.fit(X, y)

        return self.model
    
    
    def predict_(self, X, transform=True):
        
        if transform:
            X = transform_X_y(X, self.n_bodies, self.use_mass)
        
        y = self.model.predict(X)
        
        if transform:
            y = inverse_transform_X_y(y, self.n_bodies, self.use_mass)
            
        return y
        

    def model_error(self, X, y):
        
        return np.sqrt(mean_squared_error(y, self.predict_(X, transform=False)))


def get_model_data(path, steps, use_mass=False):
    
    df = pd.read_csv(path)
    
    X, y = get_X_y(df, steps, use_mass, transform=True)
    return train_test_split(X, y)

def build_model(path, step, gs_params):
    
    save_path = path.split('/')[-1].split('.')[0] + f'_steps_{step}.pkl'
    n_bodies = int(path.split('/')[-1].split('_')[1])
            
    X_train, X_test, y_train, y_test = get_model_data(path, step)
    print('Fitting model from: ', path)
    model = GSModel(n_bodies, step, gs_params)
    model.fit(X_train, y_train)
            
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
                
    print('Fit model: ', save_path, '    model error: ', model.model_error(X_test, y_test))

def main():
    
    gs_params = {
                'estimator__n_estimators': [1000],
                'estimator__max_depth': [10],
                'estimator__learning_rate': [0.05]
            }
    
    steps = [10, 100, 1000]
    paths = ['../data/samples_2_bodies_3_dim_1_m_com.csv',
             '../data/samples_3_bodies_3_dim_1_m_com.csv']
    
    processes = []
    for step in steps:
        for path in paths:
            p = mp.Process(target=build_model, args=(path, step, gs_params))
            processes.append(p)
            p.start()
    
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()

        