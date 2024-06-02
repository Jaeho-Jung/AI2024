import numpy as np


class LinearRegression():
    def __init__(self, alpha=1e-2, max_iter=100, tol=1e-3, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, batch_size=100):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.batch_size = batch_size

        self.w = None
        self.b = None

    def _initialize_parameters(self, d):
        self.w = np.zeros(d)
        self.b = 0

    def _compute_loss(self, X, y):
        y_pred = np.dot(self.w.T, X) + self.b
        error = y_pred - y
        loss = np.mean(error**2)
        return loss

    def fit(self, X, y):
        # Initialize
        # n_features: number of features
        # n_samples: number of samples
        n_features, n_samples = X.shape
        self._initialize_parameters(n_features)
        
        self.batch_size = min(self.batch_size, n_samples)

        best_loss = np.inf
        no_improvement_count = 0
        
        # Split the data
        if self.early_stopping:
            val_size = int(self.validation_fraction * n_samples)
            indices = np.random.permutation(n_samples)
            X_train, y_train = X[:,indices[val_size:]], y[indices[val_size:]]
            X_val,   y_val   = X[:,indices[:val_size]], y[indices[:val_size]]
        else:
            X_train, y_train = X, y
        
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
                dw = 2 * np.dot(error, X_batch.T) / len(y_batch)
                db = 2 * np.sum(error) / len(y_batch)
                
                # Update weights, biases
                self.w -= self.alpha * dw   
                self.b -= self.alpha * db

            # Print status
            train_loss = self._compute_loss(X_train, y_train)
            print(f"Iteration {epoch + 1}/{self.max_iter}: Training Loss = {train_loss}")

            # Early stopping
            if self.early_stopping:
                val_loss = self._compute_loss(X_val, y_val)
                print(f"Validation Loss = {train_loss}")
                
                if val_loss < best_loss - self.tol:
                    best_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= self.n_iter_no_change:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"Training finished after {epoch + 1} iterations")

    def predict(self, X):
        y_pred = np.dot(self.w.T, X) + self.b
        return y_pred