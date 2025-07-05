import numpy as np

def extract_LU(A_lu):
    """
    Given in-place LU matrix (no pivoting), extract L and U for verification.
    L has ones on the diagonal, U is upper triangular.
    """
    n = A_lu.shape[0]
    L = np.tril(A_lu, k=-1) + np.eye(n)
    U = np.triu(A_lu)
    return L, U

# Example usage:
if __name__ == "__main__":
    A_lu = np.array([[2., -1., 3.],
                     [2., 2., -2.5],
                     [-1., 0.5, 3.5]])
    L, U = extract_LU(A_lu)
    print("L =\n", L)
    print("U =\n", U)
