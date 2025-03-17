from model_gpr import adaptive_elbo
from model_gpr import adaptive_sparse_gpr
import torch
import gpytorch
import numpy as np

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import math

import matplotlib.pyplot as plt
import copy
import pickle
import time


def generate_data(num_init = None):

    X_tot = torch.linspace(0, 5,400)
    Y_tot = torch.sin(X_tot)

    if num_init is None:
        num_init = 100
    X_init = X_tot[:num_init]
    Y_init = Y_tot[:num_init]

    X_t = X_tot[num_init:]
    Y_t = Y_tot[num_init:]

    return X_init, Y_init, X_t, Y_t

def run_asgpr():
    M = 10
    num_init = 100
    num_steps_init = 50

    X_init, Y_init, X_t, Y_t = generate_data(num_init)
    X_plot = torch.cat((X_init, X_t))
    Y_plot = torch.cat((Y_init, Y_t))


    kernel = gp.kernels.RBF(input_dim=1)
    Xu = torch.linspace(0,5, M)
    lamb_ = 0.98

    # Define the model
    osgpr = adaptive_sparse_gpr.AdaptiveSparseGPRegression(
        X_init, Y_init, kernel=kernel,
        lamb=lamb_, Xu=Xu, jitter = 1e-3
    )

    # Initialize the model
    osgpr.batch_update(num_steps=num_steps_init)


    for t, _ in enumerate(zip(X_t, Y_t)):
        current_time = time.time()
        X_t.requires_grad_(True)
        X_new = X_t[t:t+1]
        Y_new = Y_t[t:t+1]
        osgpr.fast_online_update(X_new, Y_new, verbose=True)
        print('Computation time: ' + str(time.time() - current_time))

    X_plot.requires_grad_(True)
    pred, var = osgpr(X_plot, noiseless = True)
    dpred = torch.autograd.grad(pred.sum(), X_plot,
                                create_graph=True)[0]

    X_plot.requires_grad_(False)
    plt.plot( X_plot.numpy(), pred.detach().numpy(), 'red', label='mean')
    plt.plot(X_plot.numpy(), Y_plot.numpy(), 'b', label='truth',linestyle='dashed')
    plt.plot(X_plot.numpy(), dpred.detach().numpy(), 'blue', label='derivative')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_asgpr()