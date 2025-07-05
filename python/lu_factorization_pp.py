import numpy as np

def lu_factorization_pp(A):
    """
    LU factorization with partial pivoting for a square matrix A.
    Returns P, L, U such that P @ A = L @ U.
    """
    A = A.copy()
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n):
        # Partial pivoting: Find index of max element in column k, rows k:n
        max_row = np.argmax(abs(U[k:, k])) + k
        if U[max_row, k] == 0:
            raise ValueError("Matrix is singular!")

        # Swap rows in U, P, L (for L, only columns before k)
        if max_row != k:
            U[[k, max_row], :] = U[[max_row, k], :]
            P[[k, max_row], :] = P[[max_row, k], :]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        # Elimination process
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, :] = U[i, :] - L[i, k] * U[k, :]

    np.fill_diagonal(L, 1.0)  # Diagonal of L is 1

    return P, L, U

if __name__ == "__main__":
    # Example usage
    A = np.array([[2, -1, 3],
                  [4,  2, 1],
                  [-2, 1, 2]], dtype=float)
    P, L, U = lu_factorization_pp(A)
    print("P =\n", P)
    print("L =\n", L)
    print("U =\n", U)
    # Check: P @ A == L @ U
    print("Check P @ A:\n", P @ A)
    print("Check L @ U:\n", L @ U)
    print("Difference:\n", (P @ A) - (L @ U))
