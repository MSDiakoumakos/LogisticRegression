import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionEP34:
    def __init__(self, lr=1e-2):
        self.w = None
        self.b = None
        self.lr = lr
        
        # Additional attributes
        self.N = None  # number of samples
        self.p = None  # number of features
        self.f = None  # predictions during training
        self.l_grad_w = None  # gradient w.r.t. weights
        self.l_grad_b = None  # gradient w.r.t. bias
        
    def init_parameters(self):
        """Initialize parameters using Gaussian distribution"""
        self.w = np.random.randn(self.p) * 0.1
        self.b = np.random.randn() * 0.1
        
    def forward(self, X):
        """Compute p(1|x) for all samples and store result"""
        z = X @ self.w + self.b
        self.f = 1 / (1 + np.exp(-z))
        
    def predict(self, X):
        """Compute p(1|x) for all samples without storing"""
        z = X @ self.w + self.b
        return 1 / (1 + np.exp(-z))
        
    def loss(self, X, y):
        """Compute binary cross-entropy loss"""
        predictions = self.predict(X)
        return -np.mean(y * np.log(predictions + 1e-15) + 
                       (1 - y) * np.log(1 - predictions + 1e-15))
        
    def backward(self, X, y):
        """Compute gradients w.r.t. parameters"""
        # Compute error term (y - p_model)
        error = y - self.f
        
        # Compute gradients using vectorized operations
        self.l_grad_w = -(1/self.N) * (X.T @ error)
        self.l_grad_b = -np.mean(error)
        
    def step(self):
        """Perform one step of gradient descent"""
        self.w -= self.lr * self.l_grad_w
        self.b -= self.lr * self.l_grad_b
        
    def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000, show_line=False):
        """Train the model using gradient descent"""
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
            
        # Store dimensions
        self.N, self.p = X.shape
        
        # Initialize parameters
        self.init_parameters()
        
        # Create indices for shuffling
        indices = np.arange(self.N)
        
        for i in range(iterations):
            # Shuffle data at the start of each epoch
            if batch_size is None or (i % (self.N // batch_size) == 0):
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            
            # Select batch
            if batch_size is None:
                X_batch = X_shuffled
                y_batch = y_shuffled
            else:
                start_idx = (i * batch_size) % self.N
                end_idx = min(start_idx + batch_size, self.N)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            self.forward(X_batch)
            
            # Backward pass
            self.backward(X_batch, y_batch)
            
            # Update parameters
            self.step()
            
            # Show progress if needed
            if i % show_step == 0:
                current_loss = self.loss(X, y)
                print(f"Iteration {i}, Loss: {current_loss:.4f}")
                if show_line:
                    self.show_line(X, y)
    
    def show_line(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot data points for two classes, as well as the line
        corresponding to the model.
        """
        if (X.shape[1] != 2):
            print("Not plotting: Data is not 2-dimensional")
            return
        idx0 = (y == 0)
        idx1 = (y == 1)
        X0 = X[idx0, :2]
        X1 = X[idx1, :2]
        plt.plot(X0[:, 0], X0[:, 1], 'gx')
        plt.plot(X1[:, 0], X1[:, 1], 'ro')
        min_x = np.min(X, axis=0)
        max_x = np.max(X, axis=0)
        xline = np.arange(min_x[0], max_x[0], (max_x[0] - min_x[0]) / 100)
        yline = (self.w[0]*xline + self.b) / (-self.w[1])
        plt.plot(xline, yline, 'b')
        plt.show()
