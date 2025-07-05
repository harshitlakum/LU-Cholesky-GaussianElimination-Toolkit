function A_lu = lu_factorization_inplace(A)
%LU_FACTORIZATION_INPLACE Performs in-place LU factorization without pivoting.
%   After execution:
%     - Upper triangle (including diagonal) of A contains U,
%     - Strictly lower triangle contains L (with 1s implied on the diagonal).

    n = size(A,1);
    for k = 1:n-1
        if A(k,k) == 0
            error('Zero pivot encountered at row %d.', k);
        end
        % Compute multipliers for all rows below the pivot
        A(k+1:n, k) = A(k+1:n, k) / A(k, k);
        % Outer product elimination update
        A(k+1:n, k+1:n) = A(k+1:n, k+1:n) - A(k+1:n, k) * A(k, k+1:n);
    end
    A_lu = A;
end

function [L, U] = extract_LU(A_lu)
%EXTRACT_LU Extracts L and U from in-place LU matrix
    n = size(A_lu,1);
    L = tril(A_lu, -1) + eye(n);
    U = triu(A_lu);
end

% Example usage:
% A = [2 -1 3; 4 2 1; -2 1 2];
% A_lu = lu_factorization_inplace(A);
% [L, U] = extract_LU(A_lu);
% disp('A after in-place LU:');
% disp(A_lu);
% disp('L =');
% disp(L);
% disp('U =');
% disp(U);
% disp('Check L*U =');
% disp(L*U);
% disp('Original A =');
% disp(A);
% disp('Difference:');
% disp(L*U - A);
