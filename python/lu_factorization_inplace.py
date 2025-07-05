import numpy as np

def lu_factorization_inplace(A):
    """
    In-place LU factorization of A without pivoting, using outer product update and a single for-loop.
    After execution:
        - The upper triangle (including diagonal) of A contains U,
        - The subdiagonal part of A contains L (with implied 1s on the diagonal).
    """
    A = A.copy()  # To avoid modifying input outside
    n = A.shape[0]
    for k in range(n-1):
        if A[k, k] == 0:
            raise ValueError(f"Zero pivot encountered at row {k}.")
        # Compute the multipliers (l_ik) for all rows below row k
        A[k+1:, k] = A[k+1:, k] / A[k, k]
        # Outer product elimination: Update submatrix in one go
        A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])
    return A

def extract_LU(A_lu):
    """
    Given in-place LU matrix, extract L and U for verification.
    """
    n = A_lu.shape[0]
    L = np.tril(A_lu, k=-1) + np.eye(n)
    U = np.triu(A_lu)
    return L, U

if __name__ == "__main__":
    # Example usage
    A = np.array([[2., -1., 3.],
                  [4.,  2., 1.],
                  [-2., 1., 2.]])
    A_lu = lu_factorization_inplace(A)
    L, U = extract_LU(A_lu)
    print("In-place A after LU:\n", A_lu)
    print("L =\n", L)
    print("U =\n", U)
    # Check A ≈ L @ U
    print("Reconstructed A =\n", L @ U)
    print("Original A =\n", A)
    print("Difference =\n", (L @ U) - A)
