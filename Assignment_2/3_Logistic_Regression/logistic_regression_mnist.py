from sklearn import datasets
from logistic_regression import LogisticRegression
import numpy as np

X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.to_numpy()
y = y.to_numpy()

print(X.shape)
print(y.shape)

X = X.astype(np.float32)
X /= 255.
X -= X.mean(axis=0)

y = y.astype(np.int16)

model = LogisticRegression(alpha=0.001, max_iter=100, early_stopping=True)
model.fit(X.T, y)