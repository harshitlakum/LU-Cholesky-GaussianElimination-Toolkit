function [L, U, P] = lu_factorization_pp(A)
% LU_FACTORISATION_PP Computes the LU factorization with partial pivoting.
%   [L, U, P] = lu_factorization_pp(A)
%   returns lower-triangular L, upper-triangular U, and permutation matrix P
%   such that P*A = L*U.

    n = size(A, 1);
    U = A;
    L = eye(n);
    P = eye(n);

    for k = 1:n-1
        % Partial Pivoting: find max in column k below (and at) row k
        [~, m] = max(abs(U(k:n, k)));
        m = m + k - 1; % actual row index

        if m ~= k
            % Swap rows in U
            U([k m], :) = U([m k], :);
            % Swap rows in P
            P([k m], :) = P([m k], :);
            % Swap rows in L (first k-1 columns only)
            if k > 1
                L([k m], 1:k-1) = L([m k], 1:k-1);
            end
        end

        % Elimination
        for i = k+1:n
            L(i, k) = U(i, k) / U(k, k);
            U(i, :) = U(i, :) - L(i, k) * U(k, :);
        end
    end
end

% Example usage
% Uncomment below to test the function:
% A = [2 -1 3; 4 2 1; -2 1 2];
% [L, U, P] = lu_factorization_pp(A)
% % Verify: norm(P*A - L*U)

