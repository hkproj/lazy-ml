import torch

class LinearRegressionGradientDescent:
    def __init__(self, x, y, learning_rate: float, device: torch.device):
        self.x = x.to(device)
        self.y = y.to(device)
        self.learning_rate = learning_rate

        assert x.shape[0] == y.shape[0], "The number of samples in x_train and y_train must be equal"

        self.num_samples = y.shape[0]
        self.num_features = x.shape[1]

        self.bias = torch.zeros((1, 1)).to(device)
        self.weights = torch.zeros((self.num_features, 1)).to(device)

    def predict(self, bias, weights, x):
        return torch.matmul(x, weights) + bias

    def loss(self, y, predicted):
        # MSE loss
        return torch.mean(torch.square(y - predicted))
    
    def step(self):
        # Make predictions using the current weights and bias
        predicted = self.predict(self.bias, self.weights, self.x)
        # Get the loss 
        loss = self.loss(self.y, predicted)
        # Calculate the gradients
        delta_weights = 1 / self.num_samples * torch.matmul(self.x.T, (predicted - self.y.reshape(-1, 1)))
        delta_bias = 1 / self.num_samples * torch.sum(predicted - self.y.reshape(-1, 1))
        # Update the weights and bias
        self.weights = self.weights - self.learning_rate * delta_weights
        self.bias = self.bias - self.learning_rate * delta_bias
        return loss