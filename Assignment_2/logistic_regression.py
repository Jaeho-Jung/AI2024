import numpy as np


class LogisticRegression():
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

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        ret = exp_z / np.sum(exp_z, axis=0)
        
        return ret

    def one_hot_encode(self, target):
        num = np.unique(target, axis=0)
        num = num.shape[0]

        one_hot = np.eye(num)[target].T

        return one_hot

    def _compute_loss(self, X, y):
        y_pred = np.dot(self.w, X) + self.b[:, np.newaxis]
        probs = self.softmax(y_pred)
        y_one_hot = self.one_hot_encode(y)
        loss = -np.sum(np.log(probs) * y_one_hot) / X.shape[1]
        return loss
    
    def _compute_accuracy(self, X, y):
        y_pred = np.dot(self.w, X) + self.b[:, np.newaxis]
        probs = self.softmax(y_pred)
        predictions = np.argmax(probs, axis=0)
        accuracy = np.mean(predictions == y)
        return accuracy

    def fit(self, X, y):
        # Initialize
        # n_features: number of features
        # n_samples: number of samples
        # n_classes: number of classes
        n_features, n_samples = X.shape
        n_classes = int(np.max(y))+1

        self.w = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)
        self.batch_size = min(self.batch_size, n_samples)
        
        # Split the data into training and validation sets if searly stopping is enabled
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
                y_pred = np.dot(self.w, X_batch) + self.b[:, np.newaxis]
                probs = self.softmax(y_pred)
                y_one_hot = self.one_hot_encode(y_batch)
                error = probs - y_one_hot
                
                # Compute gradients
                dw = np.dot(error, X_batch.T) / len(y_batch)
                db = np.sum(error) / len(y_batch)
                
                # Update weights
                self.w -= self.alpha * dw
                self.b -= self.alpha * db

            # Print status
            train_loss = self._compute_loss(X_train, y_train)
            train_accuracy = self._compute_accuracy(X_train, y_train)
            print(f"Iteration {epoch + 1}/{self.max_iter}: Loss = {train_loss}, Training Accuracy = {train_accuracy}")
          
            # Early stopping check
            if self.early_stopping:
                val_loss = self._compute_loss(X_val, y_val)
                val_accuracy = self._compute_accuracy(X_val, y_val)
                print(f"Validation Loss = {val_loss}, Validation Accuracy = {val_accuracy}")
                
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