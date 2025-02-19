from models import StochasticVariationalGPModel
import torch
import gpytorch
import math
import numpy as np


import matplotlib.pyplot as plt

def main():

    training_iter = 50
    lb, ub = -5, 5

    number_epochs = 10
    training_iter = 50

    # Create training data
    train_x = torch.linspace(lb, ub,100)
    train_y = torch.cos(-train_x) + torch.randn(train_x.size())*math.sqrt(0.04)

    m = 20
    inducing_points = torch.zeros((m,))

    # Split into 4 equal batches
    train_x_batches = torch.split(train_x, number_epochs)
    train_y_batches = torch.split(train_y, number_epochs)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3))
    model = StochasticVariationalGPModel(inducing_points)

    # Use the adam optimizer
    # Set the learning rate lr to 0.1
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()},
         {'params': likelihood.parameters()}],
        lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    # for i in range(number_epochs):
    #     for j in range(training_iter):
    #         optimizer.zero_grad()
    #         output = model(train_x_batches[i])
    #         loss = -mll(output, train_y_batches[i])
    #         loss.backward()
    #         optimizer.step()
    #         print('Iter: {}, Loss: {}'.format(j, loss))

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Test points are regularly spaced along [0, 1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(lb,ub,100)
        observed_pred = likelihood(model(test_x))

    with torch.no_grad():
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
    plt.show()

if __name__ == '__main__':
    main()