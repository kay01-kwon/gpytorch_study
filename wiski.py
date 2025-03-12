import math
import numpy as np
import torch
import gpytorch

import matplotlib.pyplot as plt
import seaborn as sns

from online_gp import models
from online_gp.models.stems import Identity
from online_gp.utils.cuda import try_cuda

import time

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
print('The version of torch: ')
print(torch.__version__)

def get_data(num_rows, fn='cos'):

    # Generate input data
    X = torch.linspace(-5, 5, num_rows)

    if fn == 'sine':
        Y = torch.sin(X) + 0.01*torch.randn(X.shape)
    elif fn == 'cos':
        Y = torch.cos(X) + 0.01*torch.randn(X.shape)

    X, Y = try_cuda(X,Y)
    X, Y = X.view(-1,1), Y.view(-1,1)
    X, Y = X[:num_rows], Y[:num_rows]

    return X, Y

def preprocess_data(X, Y, num_init):

    init_x, X = X[:num_init], X[num_init:]
    init_y, Y = Y[:num_init], Y[num_init:]

    return init_x, init_y, X, Y

def draw_plot(model,
              train_x, train_y,
              test_x, test_y,
              inducing_x, inducing_f,
              ax,
              show_legend=False):

    inducing_x_cpu = inducing_x.detach().cpu()
    inducing_f_cpu = inducing_f.detach().cpu()

    train_x_cpu = train_x.detach().cpu().squeeze(-1)
    train_y_cpu = train_y.detach().cpu().squeeze(-1)

    test_x_cpu = test_x.cpu().squeeze(-1)

    x_min = min(train_x_cpu.min(), test_x_cpu.min(),
                inducing_x_cpu.min())

    x_max = max(train_x_cpu.max(), test_x_cpu.max(),
                inducing_x_cpu.max())

    xlim = (x_min - 1e-1, x_max + 1e-1)

    x_grid = try_cuda(torch.linspace(*xlim, 200))
    x_grid_cpu = x_grid.cpu()
    model.eval()

    with torch.no_grad():
        mean, var = model.predict(x_grid)
        lb, ub = (mean - 2*var.sqrt(),
                  mean + 2*var.sqrt())

    mean_cpu = mean.cpu().view(-1)
    lb_cpu, ub_cpu = lb.cpu().view(-1), ub.cpu().view(-1)

    ax.plot(x_grid_cpu, mean_cpu, linewidth=2,
            color='red')
    ax.fill_between(x_grid_cpu, lb_cpu, ub_cpu,
                    alpha=0.2, color='blue')
    ax.scatter(train_x_cpu, train_y_cpu,
               color='black', s=32, edgecolor='none',
               label='obs')
    ax.scatter(inducing_x_cpu, inducing_f_cpu,
               color='red', marker='+', linewidth=3,
               s=128, label='m')

    ax.set_xlim((-5.1,5.1))
    ax.set_xlabel('x')
    ax.set_ylim((-3,3))
    plt.tight_layout()
    return ax

def run_wiski():
    chunk_size = 40
    X, Y = get_data(num_rows=4*chunk_size - 1)

    init_x, init_y, X, Y = preprocess_data(X, Y,
                                           chunk_size)

    stem = Identity(input_dim = 1)
    covar_module = (gpytorch.
                  kernels.
                  SpectralMixtureKernel(
                    num_mixtures=3
                    )
                  )

    # Instantiate wiski model
    wiski_model = models.OnlineSKIRegression(stem,
    init_x, init_y, lr=1e-1, grid_size=50,
    grid_bound=10, covar_module = covar_module)
    wiski_model = try_cuda(wiski_model)
    # Pretrain the model
    wiski_model.fit(init_x, init_y, 200)

    fig = plt.figure(figsize=(15,4))
    subplot_count = 1
    wiski_model.set_lr(1e-2)

    for t, (x, y) in enumerate(zip(X, Y)):
        start_time = time.time()
        wiski_model.update(x,y)
        print('Time : %s seconds' %(time.time() - start_time))
        if t % chunk_size == 0:
            inducing_x = (wiski_model
                          .gp
                          .covar_module
                          .grid[0][1:-1])
            inducing_f, _ = wiski_model.predict(inducing_x)

            train_x = torch.cat([init_x, X[:t+1]])
            train_y = torch.cat([init_y, Y[:t+1]])
            ax = fig.add_subplot(1,3,subplot_count)

            if subplot_count == 1:
                ax.set_ylabel('y', rotation=0)
            subplot_count += 1
            ax = draw_plot(wiski_model,
                           train_x, train_y,
                           X[t+1:], Y[t+1:],
                           inducing_x, inducing_f, ax)
    plt.subplots_adjust(top=0.8)
    plt.show()

if __name__ == '__main__':
    run_wiski()

