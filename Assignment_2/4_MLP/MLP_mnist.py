from sklearn import datasets
from MLP import MLP
import numpy as np


# load datasets
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.to_numpy()
y = y.to_numpy()

print(X.shape)
print(y.shape)


# preprocess
X = X.astype(np.float32)
X /= 255.
X -= X.mean(axis=0)

y = y.astype(np.int16)


# train and save models
L = [1, 2, 3, 4]
for l in L:
    hidden_sizes = [30] * l
    model = MLP(hidden_sizes=hidden_sizes, alpha=0.001, max_iter=100, early_stopping=True)
    model.fit(X.T, y)

    model.save_model(f'MLP_mnist_{l}.pkl')