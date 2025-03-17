from model_gpr import adaptive_elbo
from model_gpr import adaptive_sparse_gpr
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import copy
import time
import sys
import pickle


import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

import time


pyro.set_rng_seed(0)
plt.rcParams['figure.dpi'] = 100

def generate_data(seed):

    pyro.set_rng_seed(seed)

    N1 = 300
    N2 = 201

    std_noise = 0.2
    X1 = torch.linspace(0.0, 3., N1)
    X2 = torch.linspace(3.0, 5., N2)

    A = torch.linspace(0.5, 2., N1)
    y1 = A * torch.sin(4.* X1) + dist.Normal(0.0, std_noise).sample(sample_shape=(N1,))
    y2 = 2 * torch.sin(8.* X2) + dist.Normal(0.0, std_noise).sample(sample_shape=(N2,))

    X = torch.concat((X1,X2))
    y = torch.concat((y1,y2))

    # Data for initial learning
    T = 100
    X_init = X[:T]
    y_init = y[:T]

    # Data for on-line update
    X_t = X[T:]
    y_t = y[T:]

    return X_init, y_init, X_t, y_t

X_init, y_init, X_t, y_t = generate_data(0)
X_plot = torch.cat((X_init,X_t))
y_plot = torch.cat((y_init,y_t))


plt.show()

# For the paper experiments
#N_iter = 1000 # Number of iterations, for each iteration a different dataset (seed) is generated

# For debugging
N_iter = 2 # Number of iterations, for each iteration a different dataset (seed) is generated

M = 20
T = 100

# Optimization parameters
num_steps_init = 200
num_steps_online = 1

lamb_ = 0.98

mse_pred_iter = []
mean_pred_iter = []
std_pred_iter = []
IC_95_iter = []
train_time_iter = []
test_time_iter = []

for seed in range(N_iter):
    print(seed)
    X_init, y_init, X_t, y_t = generate_data(seed)
    # initialize the kernel and model
    pyro.clear_param_store()
    kernel = gp.kernels.RBF(input_dim=1)

    # initialize the inducing inputs in interval [0,1]
    M = 10
    inducing_points = torch.linspace(0, 1, M)
    Xu = torch.Tensor(copy.copy(inducing_points))

    ###################################
    # Define the model
    osgpr = adaptive_sparse_gpr.AdaptiveSparseGPRegression(X_init, y_init, kernel, Xu=Xu, lamb=lamb_, jitter=1.0e-3)

    # Initialize the model
    osgpr.batch_update(num_steps=num_steps_init)
    ####################################

    mse_pred = []
    mean_pred = []
    std_pred = []
    IC_95 = []
    test_time = 0
    train_time = 0

    discrete_train_time = []

    for t, (x, y) in enumerate(zip(X_t, y_t)):
        current_time = time.time()
        X_new = X_t[t:t + 1]
        y_new = y_t[t:t + 1]

        start = time.process_time()
        # Compute test error predicting next sample
        with torch.no_grad():
            pred, cov = osgpr(X_new, noiseless=True)

        test_time += (time.process_time() - start)

        mean_pred.append(pred.numpy())

        mse = (pred - y_new) ** 2
        mse_pred.append(mse.numpy())

        std = torch.sqrt(cov)
        std_pred.append(std.numpy())
        IC_95.append((torch.abs(y_new - pred) < 2 * std).numpy())
        #####################################
        # Update model
        start = time.process_time()
        loss = osgpr.fast_online_update(X_new, y_new, L=T, M=M, perc_th=0.01)
        train_time += (time.process_time() - start)
        ######################################
        discrete_train_time.append(time.time() - current_time)
        # print('Time: ', time.process_time() - start)
    print('Maximum training time:', np.max(discrete_train_time))
    print('Minimum training time:', np.min(discrete_train_time))

    print(np.max(discrete_train_time))
    print(train_time)
    # Save variables
    mse_pred_iter.append(mse_pred)
    std_pred_iter.append(std_pred)
    mean_pred_iter.append(mean_pred)
    IC_95_iter.append(IC_95)
    train_time_iter.append(train_time)
    test_time_iter.append(test_time)

data = {'mse': mse_pred_iter,
        'std': std_pred_iter,
        'mean': mean_pred_iter,
        'IC_95': IC_95_iter,
        'train_time': train_time_iter,
        'test_time': test_time_iter}

with open('../results/Toy_Fast-AGP.pickle', 'wb') as handle:
  pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../results/Toy_Fast-AGP.pickle', 'rb') as handle:
    data_fast_AGP = pickle.load(handle)

fig, axs = plt.subplots(figsize=(12, 8))


plt.plot(X_t.numpy(), y_t.numpy(),color="black", label='Mean real signal')
# plt.plot(X_t.numpy(), y_t.numpy(),color="lightgray", marker=".", label='Data')

data = data_fast_AGP
mean = np.mean(np.squeeze(np.array(data['mean'])),0)
sd = np.mean(np.squeeze(np.array(data['std'])),0)

plt.plot(X_t, mean, color="red", label='Predicitve mean')
plt.fill_between(X_t, (mean - 2.0 * sd), (mean + 2.0 * sd), color="C0", alpha=0.3, label='Two-sigma uncertainty')
plt.legend(bbox_to_anchor=(0.9, 1.4), loc='upper left', borderaxespad=0.)
plt.xlim(1.,5.)
plt.ylim(-3.,4.)
plt.title('Fast-AGP',y=0.85)
plt.show()

plt.hist(discrete_train_time, bins=20, edgecolor='black')

plt.show()