import numpy as np

def back_substitution(U, b):
    """
    Solves Ux = b for x, where U is an upper triangular matrix.
    """
    n = U.shape[0]
    # Check input dimensions
    if U.shape[1] != n:
        raise ValueError("Matrix U must be square.")
    if b.size != n:
        raise ValueError("Vector b size must match U dimension.")
    # Check if U is upper triangular
    if not np.allclose(U, np.triu(U)):
        raise ValueError("Matrix U must be upper triangular.")

    x = np.zeros_like(b, dtype=np.double)
    # Back substitution
    for i in range(n-1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Zero diagonal element encountered!")
        sum_ = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (b[i] - sum_) / U[i, i]
    return x

if __name__ == "__main__":
    # Example
    U = np.array([[2, -1, 3],
                  [0, 1,  4],
                  [0, 0, -2]], dtype=np.double)
    b = np.array([5, 6, -4], dtype=np.double)
    x = back_substitution(U, b)
    print("Solution x:", x)
