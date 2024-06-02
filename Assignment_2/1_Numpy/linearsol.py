import numpy as np


# Ax = b
def linear_solution(A, b):
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)

    return x

def main():
    A = np.array([[1, 2], [3, 5]])
    b = np.array([1, 2])
    
    x = linear_solution(A, b)

    print(np.allclose(np.dot(A, x), b))

if __name__ == "__main__":
    main()