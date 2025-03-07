import torch
import gpytorch
import math
from matplotlib import pyplot as plt
import numpy as np

class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives,self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def main():

    lb, ub = 0.0, 5 * math.pi
    n = 50

    train_iter = 50

    train_x = torch.linspace(lb, ub, n).unsqueeze(-1)
    train_y = torch.stack([
        torch.sin(2 * train_x) + torch.cos(train_x),
        -torch.sin(train_x) + 2 * torch.cos(2 * train_x)
    ], -1).squeeze(1)

    # Add noise to the observed value
    train_y += 0.05 * torch.randn(n,2)

    # num_tasks = 2 : Value + Derivative
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = GPModelWithDerivatives(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(train_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f  lengthscale: %.3f   noise: %.3f' % (
            i + 1, train_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    # Initialize plots
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12,6))

    model.train()
    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.max_cg_iterations(50):
        test_x = torch.linspace(lb, ub, 500)
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # Plot training data as black stars
    y1_ax.plot(train_x.detach().numpy(),
               train_y[:,0].detach().numpy(),
               'k*')
    y1_ax.plot(test_x.numpy(),
               mean[:,0].detach().numpy(),
               'b')

    y1_ax.fill_between(test_x.numpy(),
                      lower[:,0].numpy(),
                      upper[:,0].numpy(),
                      alpha=0.5
                      )
    y1_ax.grid('True')
    y1_ax.legend(['Observed value',
                  'Mean',
                  'Confidence'], loc='best')
    y1_ax.set_title('Function values')

    # Plot training data as black stars
    y2_ax.plot(train_x.detach().numpy(),
               train_y[:,1].detach().numpy(),
               'k*')
    y2_ax.plot(test_x.numpy(),
               mean[:,1].detach().numpy(),
               'b')

    y2_ax.fill_between(test_x.numpy(),
                      lower[:,1].numpy(),
                      upper[:,1].numpy(),
                      alpha=0.5
                      )
    y2_ax.grid('True')
    y2_ax.legend(['Observed Derivatives',
                  'Mean',
                  'Confidence'], loc='best')
    y2_ax.set_title('Derivatives')
    plt.show()
    # plt.savefig('gp_derivative.png', dpi=600)

if __name__ == '__main__':
    main()
