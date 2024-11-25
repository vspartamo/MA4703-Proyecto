from get_data import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class ResNet(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, integration="Euler"):
        super(ResNet, self).__init__()
        self.num_layers = num_layers
        self.delta_t = 1.0 / num_layers  # Uniform step size
        self.integration = integration

        # Proyección inicial para ajustar las dimensiones
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Capas residuales
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.activation = nn.Tanh()

        # Proyección final para obtener un único valor de salida
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward_euler(self, x, layer):
        return x + self.delta_t * self.activation(layer(x))

    def runge_kutta_4(self, x, layer):
        k1 = self.activation(layer(x))
        k2 = self.activation(layer(x + 0.5 * self.delta_t * k1))
        k3 = self.activation(layer(x + 0.5 * self.delta_t * k2))
        k4 = self.activation(layer(x + self.delta_t * k3))
        return x + (self.delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, x, return_all=False):
        # Proyecta la entrada a la dimensión oculta
        x = self.input_projection(x)

        # Bloques residuales con el método de integración elegido
        activations = [x]
        pre_activations = []
        for layer in self.layers:
            if self.integration == "Euler":
                x_next = self.forward_euler(x, layer)
            elif self.integration == "RK":
                x_next = self.runge_kutta_4(x, layer)
            pre_activations.append(layer(x))  # Pre-activations for gradient
            activations.append(x_next)
            x = x_next

        # Proyecta al espacio de salida y aplica sigmoide
        output = torch.sigmoid(self.final_layer(x))

        if return_all:
            return output, activations, pre_activations
        else:
            return output

    def compute_gradient(self, labels, activations, pre_activations):
        """
        Compute gradients for weights, biases, and state variables using backpropagation.
        """
        grad = {
            "W": torch.zeros_like(self.final_layer.weight),
            "mu": torch.zeros_like(self.final_layer.bias),
            "K": [torch.zeros_like(layer.weight) for layer in self.layers],
            "b": [torch.zeros_like(layer.bias) for layer in self.layers]
        }

        N = self.num_layers
        delta_t = self.delta_t

        # Initialize rho for the final layer
        final_activation = activations[-1]
        z = self.final_layer(final_activation)
        rho = (torch.sigmoid(z) - labels) * (torch.sigmoid(z) * (1 - torch.sigmoid(z)))

        # Gradients for final layer
        grad["W"] = rho.t().mm(final_activation)
        grad["mu"] = rho.sum(dim=0)

        # Backpropagation for intermediate layers
        p = rho.mm(self.final_layer.weight)
        for k in range(N - 1, -1, -1):
            # Compute gamma and layer gradients
            gamma = p * (1 - torch.tanh(pre_activations[k]) ** 2)
            grad["K"][k] = delta_t * gamma.t().mm(activations[k])
            grad["b"][k] = delta_t * gamma.sum(dim=0)

            # Propagate p to the previous layer
            if k > 0:
                p = p.mm(self.layers[k].weight.t()) + gamma

        return grad


# Data generation (allows selection of dataset type)
def generate_data(dataset_type="donut_2d", n_samples=1000):
    """
    Generate data based on the selected dataset type.

    Args:
        dataset_type (str): Type of dataset to generate. Options are:
                            "donut_1d", "donut_2d", "spiral_2d", "squares_2d".
        n_samples (int): Number of samples to generate.

    Returns:
        torch.Tensor: Features of shape (n_samples, n_features).
        torch.Tensor: Labels of shape (n_samples, 1).
    """
    if dataset_type == "donut_1d":
        features, labels = get_data_donut_1d(n_samples)
    elif dataset_type == "donut_2d":
        features, labels = get_data_donut_2d(n_samples)
    elif dataset_type == "spiral_2d":
        features, labels = get_data_spiral_2d(n_samples)
    elif dataset_type == "squares_2d":
        features, labels = get_data_squares_2d(n_samples)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # PyTorch expects (n_samples, n_features)
    features = features.T
    labels = labels.reshape(-1, 1)  # Ensure labels have shape (n_samples, 1)

    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# Train function without regularization
def train_model(model, data_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for features, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(features)  # Obtén solo las predicciones
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# Evaluation function
def evaluate_model(model, features, labels):
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        predictions = (predictions > 0.5).float()
        accuracy = (predictions == labels).float().mean()
        print(f"Accuracy: {accuracy.item():.4f}")
    return accuracy


# Plot decision boundary
def plot_decision_boundary(model, features, labels):
    x_min, x_max = features[:, 0].min().item() - 0.1, features[:, 0].max().item() + 0.1
    y_min, y_max = features[:, 1].min().item() - 0.1, features[:, 1].max().item() + 0.1

    # Generate the grid
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )

    # Convert the grid to PyTorch tensor
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    # Forward pass to get predictions
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, preds.numpy(), alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=labels.flatten().numpy(), edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("Decision Boundary")
    plt.show()


# Main script with dataset selection and integration method
if __name__ == "__main__":
    # Prompt user to select dataset type
    dataset_type = input("Enter dataset type (donut_1d, donut_2d, spiral_2d, squares_2d): ").strip()
    integration = input("Enter integration method (Euler, RK): ").strip()

    # Generate data
    features, labels = generate_data(dataset_type=dataset_type, n_samples=1000)
    dataset = torch.utils.data.TensorDataset(features, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Model, optimizer, and loss function
    input_dim = features.shape[1]
    model = ResNet(input_dim=input_dim, num_layers=20, hidden_dim=50, integration=integration)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # Train the model
    print(f"Training ResNet on dataset '{dataset_type}' using {integration} integration...")
    train_model(model, data_loader, optimizer, criterion, num_epochs=50)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, features, labels)

    # Plot decision boundary
    print("Plotting decision boundary...")
    plot_decision_boundary(model, features, labels)