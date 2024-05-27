import numpy as np
import pickle


def gen_random_dataset(N, d, R=10, alpha=0.1):
    # 1, True 파라미터 값의 랜덤 할당
    w = np.random.uniform(-R, R, d)
    b = np.random.uniform(-R, R)
    
    # 2. 데이터셋 생성
    X = np.random.uniform(-R, R, (d, N))
    y = np.random.normal(np.dot(w.T, X), alpha*R)

    return X, y

def split_dataset(X, y, train_ratio=0.85, dev_ratio=0.05, test_ratio=0.1):
    assert train_ratio + dev_ratio + test_ratio == 1, "The sum of the ratios must be 1"
    
    N = X.shape[1]
    indices = np.random.permutation(N)
    
    train_end = int(train_ratio * N)
    dev_end = train_end + int(dev_ratio * N)
    
    train_indices = indices[:train_end]
    dev_indices = indices[train_end:dev_end]
    test_indices = indices[dev_end:]
    
    X_train, y_train = X[:, train_indices], y[train_indices]
    X_dev, y_dev = X[:, dev_indices], y[dev_indices]
    X_test, y_test = X[:, test_indices], y[test_indices]
    
    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)

def save_dataset(filename, *datasets):
    with open(filename, 'wb') as f:
        pickle.dump(datasets, f)


if __name__ == "__main__":
    Ns = [1000, 10000, 100000]
    i = 1
    for N in Ns:
        X, y = gen_random_dataset(N, 10)
        # print(X.shape)
        # print(X)
        # print()
        # print(y.shape)
        # print(y)
        train_set, dev_set, test_set = split_dataset(X, y)
        
        save_dataset(f'myrandomdataset{i}.pkl', train_set, dev_set, test_set)
        i += 1
