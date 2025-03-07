# Re-import necessary libraries due to execution state reset
import torch
import gpytorch
import matplotlib.pyplot as plt

# Define GP model for multi-dimensional input
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])  # ARD for multi-input

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Generate training data using linspace instead of random points
x1_train = torch.linspace(-2, 2, 20)  # 5 points along x1
x2_train = torch.linspace(-2, 2, 20)  # 5 points along x2

# Create 2D grid of training points
X1_train, X2_train = torch.meshgrid(x1_train, x2_train, indexing="ij")
train_x = torch.stack([X1_train.flatten(), X2_train.flatten()], dim=-1)  # Shape (25, 2)

# Define function f(x1, x2) = sin(x1) + cos(x2)
train_y = torch.sin(train_x[:, 0]) + 2*torch.sin(train_x[:, 1])  # Shape (25,)

# Define likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(train_x, train_y, likelihood)

# Set model to evaluation mode
model.eval()
likelihood.eval()

# Define test points along x1-axis (fix x2 = 0)
x1 = torch.linspace(-2, 2, 50)
fixed_x2 = torch.zeros_like(x1)  # Set x2 = 0
test_x_x1 = torch.stack([x1, fixed_x2], dim=-1)  # Shape (50, 2)
test_x_x1.requires_grad_(True)

# Compute predictions for x1-axis
pred_x1 = model(test_x_x1)

# Compute Jacobian for x1-axis
jacobian_x1 = torch.autograd.grad(pred_x1.mean.sum(), test_x_x1, create_graph=True)[0]

# Plot along x1 axis
plt.figure(figsize=(8, 5))
plt.plot(x1.numpy(), pred_x1.mean.detach().numpy(), label="GP Mean (x2=0)", color='b')
plt.plot(x1.numpy(), jacobian_x1[:, 0].detach().numpy(), label="Jacobian df/dx1 (x2=0)", linestyle='dashed', color='r')
plt.xlabel("$x_1$")
plt.ylabel("Function Value")
plt.legend()
plt.title("GP Mean and Jacobian along $x_1$ (x2=0)")
plt.grid()
plt.show()

# Define test points along x2-axis (fix x1 = 0)
x2 = torch.linspace(-2, 2, 50)
fixed_x1 = torch.zeros_like(x2)  # Set x1 = 0
test_x_x2 = torch.stack([fixed_x1, x2], dim=-1)  # Shape (50, 2)
test_x_x2.requires_grad_(True)

# Compute predictions for x2-axis
pred_x2 = model(test_x_x2)

# Compute Jacobian for x2-axis
jacobian_x2 = torch.autograd.grad(pred_x2.mean.sum(), test_x_x2, create_graph=True)[0]

# Plot along x2 axis
plt.figure(figsize=(8, 5))
plt.plot(x2.numpy(), pred_x2.mean.detach().numpy(), label="GP Mean (x1=0)", color='b')
plt.plot(x2.numpy(), jacobian_x2[:, 1].detach().numpy(), label="Jacobian df/dx2 (x1=0)", linestyle='dashed', color='r')
plt.xlabel("$x_2$")
plt.ylabel("Function Value")
plt.legend()
plt.title("GP Mean and Jacobian along $x_2$ (x1=0)")
plt.grid()
plt.show()
