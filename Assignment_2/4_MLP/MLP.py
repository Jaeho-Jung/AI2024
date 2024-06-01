import numpy as np
import pickle

class MLP():
    def __init__(self, hidden_sizes=[100], alpha=1e-4, max_iter=1000, tol=1e-3, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, batch_size=200):
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.batch_size = batch_size

        self.W = []
        self.b = [] 

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0))
        ret = exp_z / np.sum(exp_z, axis=0)
        return ret
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def one_hot_encode(self, target):
        num = np.unique(target, axis=0)
        num = num.shape[0]

        one_hot = np.eye(num)[target].T

        return one_hot

    def _compute_loss(self, X, y):
        h = X
        for i in range(len(self.W) - 1):
            z = np.dot(self.W[i], h) + self.b[i][:, np.newaxis]
            h = self.relu(z)
        o = np.dot(self.W[-1], h) + self.b[-1][:, np.newaxis]
        probs = self.softmax(o)
        y_one_hot = self.one_hot_encode(y)
        loss = -np.sum(np.log(probs) * y_one_hot) / X.shape[1]
        return loss

    def _compute_accuracy(self, X, y):
        h = X
        for i in range(len(self.W) - 1):
            z = np.dot(self.W[i], h) + self.b[i][:, np.newaxis]
            h = self.relu(z)
        o = np.dot(self.W[-1], h) + self.b[-1][:, np.newaxis]
        probs = self.softmax(o)
        predictions = np.argmax(probs, axis=0)
        accuracy = np.mean(predictions == y)
        return accuracy

    def _forward_pass(self, X):
        activations = [X]
        for i in range(len(self.W) - 1):
            z = np.dot(self.W[i], activations[-1]) + self.b[i][:, np.newaxis]
            h = self.relu(z)
            activations.append(h)
        # Output layer without ReLU
        o = np.dot(self.W[-1], activations[-1]) + self.b[-1][:, np.newaxis]
        activations.append(o)
        return activations

    def _backward_pass(self, X, y, activations):
        grads_W = [np.zeros_like(w) for w in self.W]
        grads_b = [np.zeros_like(b) for b in self.b]

        probs = self.softmax(activations[-1])
        y_one_hot = self.one_hot_encode(y)
        delta = probs - y_one_hot
        for i in range(len(self.W) - 1, -1, -1):
            grads_W[i] = np.dot(delta, activations[i].T) / X.shape[1]
            grads_b[i] = np.sum(delta, axis=1) / X.shape[1]
            if i > 0:
                delta = np.dot(self.W[i].T, delta) * (activations[i] > 0)  # Derivative through ReLU

        return grads_W, grads_b

    def fit(self, X, y):
        # Initialize
        # n_features: number of features
        # n_samples: number of samples
        # n_classes: number of classes
        n_features, n_samples = X.shape
        n_classes = int(np.max(y)) + 1

        self.W = []
        self.b = []

        # Initialize weights and biases for hidden layers
        prev_size = n_features
        for size in self.hidden_sizes:
            self.W.append(np.random.randn(size, prev_size) * np.sqrt(2 / prev_size))
            self.b.append(np.zeros(size))
            prev_size = size

        # Initialize weights and biases for output layer
        self.W.append(np.random.randn(n_classes, prev_size) * np.sqrt(2 / prev_size))
        self.b.append(np.zeros(n_classes))

        self.batch_size = min(self.batch_size, n_samples)

        if self.early_stopping:
            val_size = int(self.validation_fraction * n_samples)
            indices = np.random.permutation(n_samples)
            X_train, y_train = X[:, indices[val_size:]], y[indices[val_size:]]
            X_val,   y_val   = X[:, indices[:val_size]], y[indices[:val_size]]
        else:
            X_train, y_train = X, y

        best_loss = np.inf
        no_improvement_count = 0

        for epoch in range(self.max_iter):
            # Shuffle the data
            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled = X_train[:, permutation]
            y_train_shuffled = y_train[permutation]

            # Mini-batch gradient descent
            for i in range(0, X_train.shape[1], self.batch_size):
                X_batch = X_train_shuffled[:, i:i + self.batch_size]
                y_batch = y_train_shuffled[i:i + self.batch_size]

                # Forward pass
                activations = self._forward_pass(X_batch)
                # Backward pass
                grads_W, grads_b = self._backward_pass(X_batch, y_batch, activations)

                # Update weights
                for i in range(len(self.W)):
                    self.W[i] -= self.alpha * grads_W[i]
                    self.b[i] -= self.alpha * grads_b[i]

            train_loss = self._compute_loss(X_train, y_train)
            train_accuracy = self._compute_accuracy(X_train, y_train)
            print(f"Iteration {epoch + 1}/{self.max_iter}: Loss = {train_loss}, Training Accuracy = {train_accuracy}")

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

            grad_norm = np.linalg.norm(grads_W[-1])
            grad_norm += np.linalg.norm(grads_b[-1])
            for i in range(len(grads_W) - 1):
                grad_norm += np.linalg.norm(grads_W[i])
                grad_norm += np.linalg.norm(grads_b[i])
            if grad_norm < self.tol:
                print(f"Converged at epoch {epoch}")
                break

        print(f"Training finished after {epoch + 1} iterations")

    def save_model(self, file_path):
        model_data = {
            'weights': self.W,
            'biases': self.b
        }
        with open(file_path, 'wb') as file:
            pickle.dump(model_data, file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            model_data = pickle.load(file)
        self.W = model_data['weights']
        self.b = model_data['biases']