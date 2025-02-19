from models import ExactGPModel
import torch
import gpytorch
import math
import numpy as np


import matplotlib.pyplot as plt

def main():

    training_iter = 50

    lb, ub = -5, 5

    # Create training data
    train_x = torch.linspace(lb, ub,100)
    train_y = torch.cos(-train_x) + torch.randn(train_x.size())*math.sqrt(0.04)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-3))
    model = ExactGPModel(train_x, train_y, likelihood)

    # Use the adam optimizer
    # Set the learning rate lr to 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Output(Likelihood) from the model
        output = model(train_x)

        # Get negative log likelihood (Loss)
        nll = -mll(output, train_y)

        # Backpropagate gradients
        nll.backward()

        print('Iter %d/%d - Loss: %.3f    l: %.3f    noise: %.3f' % (
            i+1, training_iter, nll.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    for param_name, param in model.named_parameters():
        print(f'Parameter name: {param_name:42} value: {param.item()}')

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0, 1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(lb,ub,51)
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