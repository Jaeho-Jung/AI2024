from sklearn import datasets
from MLP import MLP
import numpy as np
from sklearn.metrics import accuracy_score

def split_dataset(X, y, test_ratio=0.2):
    N = X.shape[1]
    indices = np.random.permutation(N)
    
    train_ratio = 1 - test_ratio

    train_end = int(train_ratio * N)
    
    train_indices = indices[:train_end]
    test_indices = indices[train_end:]
    
    X_train, y_train = X[:, train_indices], y[train_indices]
    X_test, y_test = X[:, test_indices], y[test_indices]
    
    return (X_train, y_train), (X_test, y_test)

def main():
    # load datasets
    X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.to_numpy()
    y = y.to_numpy()

    # print(X.shape)
    # print(y.shape)

    # preprocess
    X = X.astype(np.float32)
    X /= 255.
    X -= X.mean(axis=0)
    X = X.T

    y = y.astype(np.int16)

    train_set, test_set = split_dataset(X, y)
    X_train, y_train = train_set
    X_test, y_test = test_set

    # train and save models
    L = [1, 2, 3, 4]
    for l in L:
        hidden_sizes = [32] * l
        model = MLP(hidden_sizes=hidden_sizes, alpha=0.001, max_iter=100, early_stopping=True)
        model.fit(X_train, y_train)

        model.save_model(f'MLP_mnist_{l}.pkl')

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        
        with open('accuracy_report.txt', 'a') as f:
            f.write(f'Model with {l} layers accuracy: {accuracy}\n')

if __name__ == '__main__':
    main()