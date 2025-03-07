import torch
import gpytorch
import matplotlib.pyplot as plt
from models import ExactGPModel


# Generate training data
train_x = torch.linspace(-1, 1, 5).unsqueeze(-1)  # 5 training points, 1D input
train_y = torch.sin(train_x).squeeze()

# Define likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Set model to evaluation mode
model.eval()
likelihood.eval()

# Define test points (requires gradient for Jacobian computation)
test_x = torch.linspace(-1, 1, 100).unsqueeze(-1)
test_x.requires_grad_(True)  # Important for autograd

# Compute predictive distribution **without torch.no_grad()**
pred = model(test_x)

# Compute Jacobian (d mean / d test_x)
jacobian = torch.autograd.grad(pred.mean.sum(), test_x, create_graph=True)[0]

# Plot GP Mean and its Derivative
plt.figure(figsize=(8, 5))
plt.plot(test_x.detach().numpy(), pred.mean.detach().numpy(), label="GP Mean", color='b')
plt.plot(test_x.detach().numpy(), jacobian.detach().numpy(), label="Jacobian (d Mean/dx)", color='r', linestyle='dashed')
plt.scatter(train_x.numpy(), train_y.numpy(), label="Training Data", color='black', marker='x')
plt.xlabel("x")
plt.ylabel("Value")
plt.legend()
plt.title("GP Mean and its Jacobian")
plt.grid()
plt.show()
