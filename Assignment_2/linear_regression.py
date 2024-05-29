import numpy as np

class LinearRegression():
    def __init__(self, alpha=1e-4, max_iter=1000, tol=1e-3, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, batch_size=200):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.batch_size = batch_size

        self.w = None
        self.b = None

    def fit(self, X, y):
        # Initialize
        # n_features: number of features
        # n_samples: number of samples
        n_features, n_samples = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.batch_size = min(self.batch_size, n_samples)
        
        # Split the data into training and validation sets if early stopping is enabled
        if self.early_stopping:
            val_size = int(self.validation_fraction * n_samples)
            indices = np.random.permutation(n_samples)
            X_train, y_train = X[:,indices[val_size:]], y[indices[val_size:]]
            X_val,   y_val   = X[:,indices[:val_size]], y[indices[:val_size]]
        else:
            X_train, y_train = X, y

        best_loss = np.inf
        no_improvement_count = 0
        
        for epoch in range(self.max_iter):
            # Shuffle the data
            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled = X_train[:,permutation]
            y_train_shuffled = y_train[permutation]

            # Mini-batch gradient descent
            for i in range(0, X_train.shape[1], self.batch_size):
                X_batch = X_train_shuffled[:,i:i + self.batch_size]
                y_batch = y_train_shuffled[i:i + self.batch_size]
                
                # Compute predictions
                y_pred = np.dot(self.w.T, X_batch) + self.b
                error = y_pred - y_batch
                
                # Compute gradients
                dw = (2 / len(y_batch)) * np.dot(error, X_batch.T)
                db = (2 / len(y_batch)) * np.sum(error)
                
                # Update weights
                self.w -= self.alpha * dw
                self.b -= self.alpha * db

                 # Print status after each batch
                print(f"Iteration {epoch + 1}/{self.max_iter}, Batch {i//self.batch_size + 1}/{int(np.ceil(X.shape[1] / self.batch_size))}: Loss = {np.mean(error ** 2)}")
            
            # Early stopping check
            if self.early_stopping:
                y_val_pred = np.dot(self.w.T, X_val) + self.b
                val_loss = np.mean((y_val_pred - y_val) ** 2)
                
                if val_loss > best_loss - self.tol:
                    no_improvement_count += 1
                else:
                    best_loss = val_loss
                    no_improvement_count = 0
                
                if no_improvement_count >= self.n_iter_no_change:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # TODO: 이거 필요한가?
            # Check for convergence
            if np.linalg.norm(dw) < self.tol and np.abs(db) < self.tol:
                print(f"Converged at epoch {epoch}")
                break

        print(f"Training finished after {epoch + 1} iterations")

# Example usage:
from gen_random_dataset import gen_random_dataset

X, y = gen_random_dataset(1000, 10)

model = LinearRegression(alpha=0.0001, max_iter=1000, early_stopping=True)
model.fit(X, y)