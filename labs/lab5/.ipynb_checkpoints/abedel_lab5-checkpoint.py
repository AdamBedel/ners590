# Basic Packages
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import pandas as pd
import time
import joblib #for parallelization
# Scikit-learn
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Scikit-optimise
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
# For grid tuning
from itertools import product
import multiprocessing

# Exercise Set 2 (a)
learning_rate = [1e-4, 5e-4, 7.5e-4, 1e-3]
num_layers = [3, 4, 5, 6]
num_nodes = [50, 100, 150, 200]

# Exercise Set 2 (b)
dim_learning_rate = Real(low=1e-4, high=1e-3, name='learning_rate')
dim_num_layers = Integer(low=3, high=8, name='num_layers')                  
dim_num_nodes = Categorical(categories=(25, 50, 100, 150, 200, 250, 300), \
                            name='num_nodes')

init_guess = [5e-4, 4, 100]

# Exercise Set 2 (c)
learning_rate_random = np.random.uniform(1e-4, 1e-3, size=4)
num_layers_random = np.random.randint(3, 9, size=4)     # rand over [3,8]
num_nodes_random = np.random.choice([25, 50, 100, 150, 200, 250, 300], size=4)

# Exercise Set 2 (d)
#-----------------------
# Data preprocessing
#-----------------------

#load the x,y data and convert to numpy array
xurl='https://raw.githubusercontent.com/aims-umich/ners590data/main/crx.csv'
yurl='https://raw.githubusercontent.com/aims-umich/ners590data/main/powery.csv'
xdata=pd.read_csv(xurl).values
ydata=pd.read_csv(yurl).values

# split into training/testing sets
xtrain, xtest, ytrain, ytest=train_test_split(xdata, ydata, test_size=0.2, random_state=42)

#create min-max scaled data
xscaler = MinMaxScaler()
yscaler = MinMaxScaler()
Xtrain=xscaler.fit_transform(xtrain)
Xtest=xscaler.transform(xtest)
Ytrain=yscaler.fit_transform(ytrain)
Ytest=yscaler.transform(ytest)

# Exercise Set 2 (e)
def fitness(args):
    lr, nl, nn = args
    n_nodes = np.full(nl, nn)
    model = Sequential()
    model.add(Input(shape=(Xtrain.shape[1],)))
    model.add(Dense(n_nodes[0], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    for i in range(1, nl):
        model.add(Dense(n_nodes[i], kernel_initializer='normal',activation='relu'))
    model.add(Dense(Ytrain.shape[1], kernel_initializer='normal',activation='linear'))
    
    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=lr), metrics=['mean_absolute_error'])

    model.fit(Xtrain, Ytrain, epochs=10, batch_size=8, validation_split = 0.15, verbose=False)
    Ynn=model.predict(Xtest)
    return r2_score(Ytest,Ynn)

# Exercise Set 3 (a)
configs = list(product(learning_rate, num_layers, num_nodes))
print(len(configs))

configs_random = list(product(learning_rate_random, num_layers_random, num_nodes_random))
print(len(configs_random))

# Exercise Set 3 (b)

t0 = time.time()
ncores = 8

if __name__ == "__main__":
    core_list=[]
    for item in (configs_random):
        core_list.append(item)
    p = multiprocessing.Pool(ncores)
    results = p.map(fitness, core_list)
    p.close()
    p.join()

print('Random Tuning in parallel with Pool took {}s to complete'.format(round(time.time()-t0)))