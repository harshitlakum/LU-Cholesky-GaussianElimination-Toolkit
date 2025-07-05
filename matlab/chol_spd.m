function L = chol_spd(A, tol)
% CHOL_SPD  Compute Cholesky factorization if A is SPD.
%   L = CHOL_SPD(A) returns lower-triangular L such that A = L*L'.
%   Errors if A is not symmetric or not positive definite.
%
%   L = CHOL_SPD(A, TOL) uses tolerance TOL for symmetry check.

    if nargin < 2, tol = 1e-10; end
    [n, m] = size(A);
    if n ~= m
        error('Matrix must be square.');
    end

    % 1) Check symmetry
    if max(max(abs(A - A'))) > tol
        error('Matrix is not self-adjoint (symmetric).');
    end

    % 2) Cholesky
    L = zeros(n);
    for k = 1:n
        % Compute diagonal entry
        temp = A(k,k) - L(k,1:k-1)*L(k,1:k-1)';
        if temp <= tol
            error('Matrix is not positive definite at pivot %d (%.3g)', k, temp);
        end
        L(k,k) = sqrt(temp);
        % Compute subdiagonal entries
        for i = k+1:n
            L(i,k) = (A(i,k) - L(i,1:k-1)*L(k,1:k-1)') / L(k,k);
        end
    end
end

% Example usage:
% A1 = [4 1 2; 1 2 0; 2 0 3];
% L1 = chol_spd(A1)
% disp('Check A1 ≈ L1*L1'');'), disp(L1*L1')
%
% A2 = [0 0; 0 1];   % symmetric but not PD
% try
%     chol_spd(A2);
% catch ME
%     disp(['Error on A2: ' ME.message]);
% end
%
% A3 = [1 2; 3 4];   % not symmetric
% try
%     chol_spd(A3);
% catch ME
%     disp(['Error on A3: ' ME.message]);
% end
