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
        # Calculate the gradients (derivative of the loss function w.r.t. the weights and bias)
        delta_weights = 1 / self.num_samples * torch.matmul(self.x.T, (predicted - self.y.reshape(-1, 1)))
        delta_bias = 1 / self.num_samples * torch.sum(predicted - self.y.reshape(-1, 1))
        # Update the weights and bias
        self.weights = self.weights - self.learning_rate * delta_weights
        self.bias = self.bias - self.learning_rate * delta_bias
        return loss
    
class LinearRegressionMLE:

    def __init__(self, x, y, device: torch.device):
        self.x = x.to(device)
        self.y = y.to(device)

        self.device = device
        assert x.shape[0] == y.shape[0], "The number of samples in x_train and y_train must be equal"

        self.num_samples = y.shape[0]
        self.num_features = x.shape[1]

        self.bias = None
        self.weights = None

    def train(self):
        # The model is defined as y = X * w + b
        # The formula for the weights is obtained by taking the derivative of the log likelihood function w.r.t. the weights

        # The formula for the weights is (X^T * X)^-1 * X^T * y
        inverse_xt_x = torch.inverse(torch.matmul(self.x.T, self.x))
        self.weights = torch.matmul(torch.matmul(inverse_xt_x, self.x.T), self.y.reshape(-1, 1))

        # The formulas for the bias is obtained by taking the derivative of the log likelihood function w.r.t. the bias
        # The formula for the bias is 1 / N * sum(y - X * w)
        self.bias = 1 / self.num_samples * torch.sum(self.y.reshape(-1, 1) - torch.matmul(self.x, self.weights))

    def predict(self, x):
        return torch.matmul(x.to(self.device), self.weights) + self.bias


class LinearRegressionMAP:

    def __init__(self, x, y, alpha: float, device: torch.device):
        self.x = x.to(device)
        self.y = y.to(device)

        self.device = device
        assert x.shape[0] == y.shape[0], "The number of samples in x_train and y_train must be equal"

        self.num_samples = y.shape[0]
        self.num_features = x.shape[1]

        self.bias = None
        self.weights = None

        self.alpha = alpha

    def train(self):
        # The model is defined as y = X * w + b
        # The formula for the weights is obtained by taking the derivative of the log of the posterior w.r.t. the weights

        # The formula for the weights is (X^T * X + alpha * I)^-1 * X^T * y
        inverse_xt_x = torch.inverse(torch.matmul(self.x.T, self.x) + self.alpha * torch.eye(self.num_features).to(self.device))
        self.weights = torch.matmul(torch.matmul(inverse_xt_x, self.x.T), self.y.reshape(-1, 1))

        # The formulas for the bias is obtained by taking the derivative of the log of the posterior w.r.t. the bias
        # The formula for the bias is 1 / N * sum(y - X * w)
        self.bias = 1 / self.num_samples * torch.sum(self.y.reshape(-1, 1) - torch.matmul(self.x, self.weights))

    def predict(self, x):
        return torch.matmul(x.to(self.device), self.weights) + self.bias