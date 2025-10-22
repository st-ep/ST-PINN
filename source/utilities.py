import numpy as np
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    """
    Physics-Informed Neural Network with weight normalization.

    PyTorch implementation replacing TensorFlow version.
    """

    def __init__(self, *inputs, layers, device=None):
        """
        Initialize the neural network.

        Args:
            *inputs: Variable number of input arrays for computing mean/std normalization
            layers: List of integers defining the network architecture [input_dim, hidden1, hidden2, ..., output_dim]
            device: torch device ('cuda' or 'cpu'). If None, automatically detects.
        """
        super(NeuralNet, self).__init__()

        self.layers = layers
        self.num_layers = len(self.layers)

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Compute input normalization parameters
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = torch.zeros([1, in_dim], dtype=torch.float32, device=self.device)
            self.X_std = torch.ones([1, in_dim], dtype=torch.float32, device=self.device)
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = torch.tensor(X.mean(0, keepdims=True), dtype=torch.float32, device=self.device)
            self.X_std = torch.tensor(X.std(0, keepdims=True), dtype=torch.float32, device=self.device)

        # Initialize weights, biases, and gammas
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.gammas = nn.ParameterList()

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]

            # Initialize weights with normal distribution
            W = torch.randn(in_dim, out_dim, dtype=torch.float32, device=self.device)
            b = torch.zeros(1, out_dim, dtype=torch.float32, device=self.device)
            g = torch.ones(1, out_dim, dtype=torch.float32, device=self.device)

            # Convert to parameters
            self.weights.append(nn.Parameter(W))
            self.biases.append(nn.Parameter(b))
            self.gammas.append(nn.Parameter(g))

    def forward(self, *inputs):
        """
        Forward pass through the network.

        Args:
            *inputs: Variable number of input tensors to concatenate

        Returns:
            List of output tensors (one per output dimension)
        """
        # Concatenate inputs and normalize
        H = torch.cat(inputs, 1)
        H = (H - self.X_mean) / self.X_std

        # Forward propagation through layers
        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]

            # Weight normalization
            V = W / torch.norm(W, dim=0, keepdim=True)

            # Linear transformation
            H = torch.matmul(H, V)

            # Add bias with gamma scaling
            H = g * H + b

            # Activation (tanh for hidden layers, linear for output)
            if l < self.num_layers - 2:
                H = torch.tanh(H)

        # Split output into list (for compatibility with TensorFlow version)
        Y = torch.split(H, 1, dim=1)

        return Y


def mean_squared_error(pred, exact):
    """
    Compute mean squared error between prediction and exact values.

    Args:
        pred: Predicted values (numpy array or torch tensor)
        exact: Exact/target values (numpy array or torch tensor)

    Returns:
        MSE value (float or torch tensor)
    """
    if isinstance(pred, np.ndarray):
        return np.mean(np.square(pred - exact))
    else:
        return torch.mean(torch.square(pred - exact))


def relative_error(pred, exact):
    """
    Compute relative L2 error between prediction and exact values.

    Args:
        pred: Predicted values (numpy array or torch tensor)
        exact: Exact/target values (numpy array or torch tensor)

    Returns:
        Relative L2 error (float or torch tensor)
    """
    if isinstance(pred, np.ndarray):
        return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact)))
    else:
        return torch.sqrt(torch.mean(torch.square(pred - exact)) / torch.mean(torch.square(exact)))


def get_device():
    """
    Get the default device for computation.

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed=1234):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False