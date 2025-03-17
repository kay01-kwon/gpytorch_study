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

    # Along X1, X2, and X3 axis
    X1_tot = torch.linspace(-5, 5,400)
    X2_tot = X1_tot
    X3_tot = X1_tot


    X_tot = torch.stack([X1_tot,
                         X2_tot,
                         X3_tot], dim = 1)


    Y_tot = (torch.sin(X_tot[:,0])
             + torch.sin(2*X_tot[:,1])
             + torch.sin(X_tot[:,2]))


    X_init = X_tot[:num_init,:]
    Y_init = Y_tot[:num_init]

    X_t = X_tot[num_init:, :]
    Y_t = Y_tot[num_init:]

    return X_init, Y_init, X_t, Y_t

def run_asgpr():
    M = 100
    num_init = 50
    num_steps_init = 400

    X_init, Y_init, X_t, Y_t = generate_data(num_init)


    kernel = gp.kernels.RBF(input_dim = 3)
    inducing_points = X_init[-2*M::2,:]

    print(inducing_points.size())
    Xu = torch.Tensor(copy.copy(inducing_points))

    lamb_ = 0.98


    # Define the model
    osgpr = adaptive_sparse_gpr.AdaptiveSparseGPRegression(
        X_init, Y_init, kernel=kernel,
        lamb=lamb_, Xu=Xu, jitter = 1e-4
    )

    # Initialize the model
    osgpr.batch_update(num_steps=num_steps_init)


    for t, _ in enumerate(zip(X_t, Y_t)):
        X_new = X_t[t:t+1]
        Y_new = Y_t[t:t+1]
        # print(X_new, Y_new)
        current_time = time.time()
        osgpr.fast_online_update(X_new, Y_new)
        # print('time: ', time.time()-current_time)

        if t % 100 == 0:
            X1_plot = torch.linspace(-5, 5, 200)
            X2_plot = X1_plot
            X3_plot = X1_plot

            X_plot = torch.stack([X1_plot,
                                  X2_plot,
                                  X3_plot], dim=1)

            X_plot.requires_grad_(True)
            pred, var = osgpr(X_plot, noiseless = True)
            dpred = torch.autograd.grad(pred.sum(), X_plot,
                                create_graph=True)[0]

            delta = 2

            X1_near = torch.linspace(X_new[0,0] - delta, X_new[0,0] + delta, 50)
            X2_near = torch.linspace(X_new[0, 1] - delta, X_new[0, 1] + delta, 50)
            X3_near = torch.linspace(X_new[0, 2] - delta, X_new[0, 2] + delta, 50)


            X_near = torch.stack([X1_near, X2_near, X3_near], dim=1)

            X_near.requires_grad_(True)
            pred_near, var_near = osgpr(X_near, noiseless = True)
            dpred_near = torch.autograd.grad(pred_near.sum(), X_near,
                                             create_graph=True)[0]
            std_near = torch.sqrt(var_near)

            f_taylor1 = (pred_near[0] +
                        dpred_near[0,0] * (X1_near - X1_near[0]) +
                        dpred_near[0,1] * (X2_near - X2_near[0]) +
                        dpred_near[0,2] * (X3_near - X3_near[0]))

            f_taylor2 = (pred_near[10] +
                        dpred_near[10,0] * (X1_near - X1_near[10]) +
                        dpred_near[10,1] * (X2_near - X2_near[10]) +
                        dpred_near[10,2] * (X3_near - X3_near[10]))

            f_taylor3 = (pred_near[25] +
                        dpred_near[25,0] * (X1_near - X1_near[25]) +
                        dpred_near[25,1] * (X2_near - X2_near[25]) +
                        dpred_near[25,2] * (X3_near - X3_near[25]))

            print(f_taylor1.size())

            X_plot.requires_grad_(False)
            X_near.requires_grad_(False)

            Y_true = (torch.sin(X_plot[:,0])
              + torch.sin(2*X_plot[:,1])
              + torch.sin(X_plot[:,2]))

            X_plot_1d = X_plot[:,0]

            pred_np = pred_near.detach().numpy()
            f_taylor1_np = f_taylor1.detach().numpy()
            f_taylor2_np = f_taylor2.detach().numpy()
            f_taylor3_np = f_taylor3.detach().numpy()

            dpred_np = dpred_near.detach().numpy()
            Y_true_np = Y_true.detach().numpy()
            std_np = std_near.detach().numpy()

            X1_near_np = X1_near.detach().numpy()


            plt.figure()
            plt.scatter(X_new[:,0].numpy(), Y_new.numpy(),marker='x')
            plt.scatter(X1_near_np[0], pred_np[0], marker='+')
            plt.scatter(X1_near_np[10], pred_np[10],marker='+')
            plt.scatter(X1_near_np[25], pred_np[25], marker='+')
            plt.plot(X1_near_np, f_taylor1_np,
                     color='green', label='Taylor expansion 1',
                     linestyle='--')
            plt.plot(X1_near_np, f_taylor2_np,
                     color='red', label='Taylor expansion 2',
                     linestyle='--')
            plt.plot(X1_near_np, f_taylor3_np,
                     color='blue', label='Taylor expansion 3',
                     linestyle='--')
            plt.plot(X_plot_1d, Y_true_np, color='green', label='truth')
            plt.plot(X1_near_np, pred_np, color='black', label='Prediction')
            plt.fill_between(X_near[:,0].detach().numpy(), pred_np - 2.0*std_np,
                             pred_np + 2.0*std_np,
                             alpha=0.2, color='orange')
            plt.grid(True)
            plt.legend()
            
            # plt.figure()
            # plt.plot(X_plot_1d.numpy(), dpred_np[:,0] + dpred_np[:,1] + dpred_np[:,2], 'red', label='derivative')
            # plt.plot(X_plot_1d.numpy(), Y_der_true, 'blue', label='true derivative', linestyle='dashed')
            # plt.legend()
            # plt.grid(True)
    plt.show()



    # X1_plot = torch.linspace(-5, 5,50)
    # X2_plot = torch.zeros_like(X1_plot)
    # X3_plot = torch.zeros_like(X1_plot)

    # X2_plot = torch.linspace(-5, 5,50)
    # X1_plot = torch.zeros_like(X2_plot)
    # X3_plot = torch.zeros_like(X2_plot)
    #
    # X_plot = torch.stack([X1_plot,
    #                       X2_plot,
    #                       X3_plot], dim = 1)
    #
    # X_plot.requires_grad_(True)
    # pred, var = osgpr(X_plot, noiseless = True)
    # dpred = torch.autograd.grad(pred.sum(), X_plot,
    #                             create_graph=True)[0]
    #
    # X_plot.requires_grad_(False)
    #
    # Y_true = (torch.sin(X_plot[:,0])
    #           + torch.sin(2*X_plot[:,1])
    #           + torch.sin(X_plot[:,2]))
    # # Y_true = (0.1*X_plot[:, 0] + 0*X_plot[:, 1] + 1*X_plot[:, 2])
    #
    # X_plot_1d = X_plot[:,1]
    #
    # pred_np = pred.detach().numpy()
    # dpred_np = dpred.detach().numpy()
    # Y_true_np = Y_true.detach().numpy()
    #
    # plt.plot( X_plot_1d.numpy(), pred_np, 'red', label='mean')
    # plt.plot(X_plot_1d.numpy(), Y_true_np, 'b', label='truth',linestyle='dashed')
    # plt.plot(X_plot_1d.numpy(), dpred_np[:,0], 'blue', label='derivative')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__':
    run_asgpr()