import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class StochasticVariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, likelihood):

        # Initialize Variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = (
            VariationalStrategy(self,
                                inducing_points,
                                variational_distribution,
                                learn_inducing_locations=True
                                )
        )
        super(StochasticVariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)