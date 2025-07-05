import numpy as np

def cholesky_spd(A, tol=1e-10):
    """
    Performs Cholesky factorization of A if and only if A is
    self-adjoint and positive definite.
    Returns lower-triangular L such that A = L @ L.T.conj().
    Raises ValueError if not self-adjoint, or np.linalg.LinAlgError if not positive definite.
    """
    A = np.array(A, dtype=np.complex128)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")

    # 1) Check self-adjointness
    if not np.allclose(A, A.T.conj(), atol=tol):
        raise ValueError("Matrix is not self-adjoint (Hermitian/symmetric).")

    # 2) Cholesky factorization
    L = np.zeros_like(A)
    for k in range(n):
        temp = A[k, k] - np.dot(L[k, :k], L[k, :k].conj())
        if temp.real <= tol:
            raise np.linalg.LinAlgError(
                f"Matrix is not positive definite at pivot {k}: got {temp}"
            )
        L[k, k] = np.sqrt(temp)
        for i in range(k+1, n):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], L[k, :k].conj())) / L[k, k]
    return L

if __name__ == "__main__":
    # Example 1: SPD
    A1 = np.array([[4, 1, 2],
                   [1, 2, 0],
                   [2, 0, 3]], dtype=float)
    L1 = cholesky_spd(A1)
    print("L1 =\n", L1)
    print("Check A1 ≈ L1 @ L1.T:\n", L1 @ L1.T, "\n")

    # Example 2: symmetric but not positive definite
    A2 = np.array([[0, 0],
                   [0, 1]], dtype=float)
    try:
        _ = cholesky_spd(A2)
    except Exception as e:
        print("Error on A2:", e)

    # Example 3: non-Hermitian
    A3 = np.array([[1, 2],
                   [3, 4]], dtype=float)
    try:
        _ = cholesky_spd(A3)
    except Exception as e:
        print("Error on A3:", e)
