import math
import torch
import gpytorch
import os
from matplotlib import pyplot as plt

training_iter = 50

n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))

# Use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_X, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_X, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main():

    train_X = torch.linspace(0, 1, 100)
    train_y = ( torch.sin(train_X* (2 * math.pi))
               + torch.randn(train_X.size()) * math.sqrt(0.04) )

    # Initialize likelihood and model
    likelihood = (gpytorch.
                  likelihoods.
                  GaussianLikelihood())
    model = ExactGPModel(train_X, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_X)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()


    model.eval()
    likelihood.eval()

    test_X = torch.linspace(0, 1, 51)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_X))

    with torch.no_grad():
        f, ax = plt.subplots(1,1,figsize=(4,3))

        lower, upper = observed_pred.confidence_region()

        ax.plot(train_X.numpy(), train_y.numpy(), 'k*')
        ax.plot(test_X.numpy(), observed_pred.mean.numpy(),'b')

        ax.fill_between(test_X.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed', 'Mean', 'Confidence'])
        plt.show()

if __name__ == "__main__":
    main()