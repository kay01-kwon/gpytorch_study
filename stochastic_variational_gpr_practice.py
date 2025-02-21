from models import StochasticVariationalGPModel
import torch
import gpytorch
import math
import numpy as np


import matplotlib.pyplot as plt

def main():

    training_iter = 500

    N1 = 100
    lb, ub = 0, 5

    N2 = 25
    lb2, ub2 = 5, 8

    # Create training data
    train_x = torch.linspace(lb, ub,N1)
    train_y = torch.sin(train_x) + torch.randn(train_x.size())*math.sqrt(0.04)

    # Create inducing points
    m = 20
    inducing_points = torch.zeros((m,))

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3))
    model = StochasticVariationalGPModel(inducing_points, likelihood)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    # Set the learning rate lr to 0.1
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()},
         # {'params': likelihood.parameters()}
         ],
        lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.VariationalELBO(likelihood,
                                        model,
                                        num_data=train_y.size(0))

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('Iteration: ', i, '\t Loss: ', loss.item())

    # Test points are regularly spaced along [0, 1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(lb,ub2,100)
        observed_pred = likelihood(model(test_x))

    # Initialize plot
    f, ax = plt.subplots(1,1,figsize=(16,12))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(),'k*')

    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')

    ax.fill_between(test_x.numpy(),
                    lower.numpy(),
                    upper.numpy(),
                    alpha=0.3)
    ax.legend(['Observed data',
                'Mean',
                'Confidence'])

    val_x = torch.linspace(lb2, ub2, N2)
    val_y = torch.sin(val_x) + torch.randn_like(val_x)*math.sqrt(0.04)

    # Conditioned model
    cond_model = (model
                  .variational_strategy
                  .get_fantasy_model(inputs=val_x,
                                     targets=val_y.squeeze())
                  )
    print(cond_model)

    with torch.no_grad():
        updated_posterior = (cond_model
                             .likelihood(cond_model(test_x))
                             )
    plt.plot(test_x.numpy(),
             updated_posterior.mean,
             color='red')
    plt.fill_between(test_x.numpy(), *updated_posterior.confidence_region(), alpha=0.5)
    plt.scatter(val_x, val_y, color='red', marker='*')

    plt.show()

if __name__ == '__main__':
    main()